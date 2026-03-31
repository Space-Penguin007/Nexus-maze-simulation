from controller import Robot
import math
import struct
import time
from collections import deque, defaultdict

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None


# ============================================================
#  RoboCupJunior Rescue Simulation controller
#  - tailored to the uploaded custom robot
#  - designed for Erebus / Webots remote-controller style
#  - strongest in Areas 1-3 (tile and quarter-tile worlds)
#  - includes a conservative fallback for Area 4
# ============================================================

TIME_STEP = 32
MAX_SPEED = 6.28
CELL = 0.06            # quarter tile = 6 cm
TILE = 0.12            # full tile = 12 cm
HALF_TILE_CM = 6       # rules use half tile for victim report tolerance

# Motion tuning
LINEAR_KP = 7.0
ANGULAR_KP = 5.2
ANGULAR_KD = 0.25
MAX_CRUISE = 0.65 * MAX_SPEED
MAX_TURN = 0.55 * MAX_SPEED
POSITION_TOL = 0.015   # 1.5 cm
HEADING_TOL = math.radians(6)

# Range sensor thresholds; these almost certainly need final tuning in your setup.
WALL_NEAR = 0.12
WALL_VERY_NEAR = 0.07
OPEN_SPACE = 0.20

# Stuck detection
STUCK_TIME = 3.0
STUCK_DIST = 0.01
MAX_LOCAL_RECOVERY = 3

# Game strategy
GAME_INFO_PERIOD = 1.0
VICTIM_STOP_TIME = 1.3
REPORT_COOLDOWN = 2.0
EXIT_TIME_LEFT_SIM = 25
EXIT_TIME_LEFT_REAL = 40

# Passage colors from the rules matrix:
# 1-2 blue=b, 1-3 yellow=y, 1-4 green=g,
# 2-3 purple=p, 2-4 orange=o, 3-4 red=r
PASSAGE_CODES = {
    "blue": "b",
    "yellow": "y",
    "green": "g",
    "purple": "p",
    "orange": "o",
    "red": "r",
}

DIRS = ["N", "E", "S", "W"]
DIR_VEC = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}
LEFT_OF = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT_OF = {v: k for k, v in LEFT_OF.items()}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
DIR_TO_YAW = {
    "N": -math.pi / 2,
    "E": 0.0,
    "S": math.pi / 2,
    "W": math.pi,
}


# -----------------------------
# Utility helpers
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_to_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def avg(vals, default=0.0):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else default


def now_s():
    return time.time()


class DeviceManager:
    def __init__(self, robot):
        self.robot = robot

    def get_any(self, names, enable=False, required=False):
        for name in names:
            try:
                dev = self.robot.getDevice(name)
                if dev is None:
                    continue
                if enable:
                    try:
                        dev.enable(TIME_STEP)
                    except Exception:
                        pass
                return dev
            except Exception:
                continue
        if required:
            raise RuntimeError(f"Could not find any device in: {names}")
        return None


class PoseEstimator:
    """
    Lightweight pose estimator.
    Primary source: GPS x/z.
    Heading source priority:
      1) displacement vector from GPS when translating,
      2) integrated gyro yaw rate while rotating / nearly static.
    """
    def __init__(self):
        self.x = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.last_x = None
        self.last_z = None
        self.last_t = None
        self.gyro_bias = 0.0
        self.gyro_axis = 1

    def _pick_gyro_axis(self, g):
        axis = max(range(len(g)), key=lambda i: abs(g[i]))
        self.gyro_axis = axis

    def update(self, t, gps_values, gyro_values=None):
        x, z = gps_values[0], gps_values[2]

        if self.last_t is None:
            self.x, self.z = x, z
            self.last_x, self.last_z = x, z
            self.last_t = t
            if gyro_values is not None:
                self._pick_gyro_axis(gyro_values)
            return

        dt = max(t - self.last_t, 1e-3)
        dx = x - self.last_x
        dz = z - self.last_z
        dist = math.hypot(dx, dz)

        # Prefer GPS displacement when significant.
        if dist > 0.002:
            self.yaw = math.atan2(dz, dx)
        elif gyro_values is not None:
            if len(gyro_values) >= 3:
                self._pick_gyro_axis(gyro_values)
            yaw_rate = gyro_values[self.gyro_axis] - self.gyro_bias
            self.yaw = wrap_to_pi(self.yaw + yaw_rate * dt)

        self.x, self.z = x, z
        self.last_x, self.last_z = x, z
        self.last_t = t

    def pos(self):
        return self.x, self.z

    def heading(self):
        return self.yaw

    def snap_grid(self):
        return int(round(self.x / CELL)), int(round(self.z / CELL))


class RobotIO:
    def __init__(self, robot):
        self.robot = robot
        dm = DeviceManager(robot)

        # Wheels: Erebus examples use "wheel1 motor" / "wheel2 motor"
        self.left_motor = dm.get_any(["wheel1 motor", "wheel1", "left wheel motor"], required=True)
        self.right_motor = dm.get_any(["wheel2 motor", "wheel2", "right wheel motor"], required=True)
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Custom robot sensors from the uploaded JSON plus common Erebus fallbacks
        self.ds_front_left = dm.get_any(["ds_front_left", "distance sensor1", "Distance Sensor 1", "ps5"], enable=True)
        self.ds_left = dm.get_any(["ds_left", "distance sensor2", "Distance Sensor 2", "ps7"], enable=True)
        self.ds_right = dm.get_any(["ds_right", "distance sensor3", "Distance Sensor 3", "ps0"], enable=True)
        self.ds_front_right = dm.get_any(["ds_front_right", "distance sensor4", "Distance Sensor 4", "ps2"], enable=True)
        self.ds_back = dm.get_any(["distance sensor5", "distance sensor6", "ds_back", "rear distance sensor"], enable=True)
        self.ds_front = dm.get_any(["distance sensor6", "ds_front", "front distance sensor"], enable=True)

        self.gps = dm.get_any(["gps"], enable=True, required=True)
        self.gyro = dm.get_any(["gyro"], enable=True)
        self.color = dm.get_any(["color", "colour_sensor", "color sensor", "Colour sensor"], enable=True)
        self.camera_left = dm.get_any(["camera_left", "camera2", "left camera"], enable=True)
        self.camera_right = dm.get_any(["camera_right", "camera3", "right camera"], enable=True)
        self.lidar = dm.get_any(["lidar"], enable=True)
        if self.lidar is not None:
            try:
                self.lidar.enablePointCloud()
            except Exception:
                pass

        self.emitter = dm.get_any(["emitter"], required=True)
        self.receiver = dm.get_any(["receiver"], enable=True)

        self.pose = PoseEstimator()
        self.last_game_info_req = -999
        self.score = 0.0
        self.time_left_sim = 999
        self.time_left_real = 999

    def step(self):
        return self.robot.step(TIME_STEP)

    def update_pose(self):
        g = self.gyro.getValues() if self.gyro is not None else None
        self.pose.update(self.robot.getTime(), self.gps.getValues(), g)

    def set_speed(self, left, right):
        self.left_motor.setVelocity(clamp(left, -MAX_SPEED, MAX_SPEED))
        self.right_motor.setVelocity(clamp(right, -MAX_SPEED, MAX_SPEED))

    def stop(self):
        self.set_speed(0.0, 0.0)

    def range_value(self, dev):
        if dev is None:
            return None
        try:
            v = dev.getValue()
        except Exception:
            return None

        # Erebus distance sensors usually give normalized near=low style values in examples,
        # but custom sensors can be metric-like. We normalize into a rough distance estimate.
        if v <= 1.0:
            return v
        # crude inverse-like fallback for raw proximity sensors
        return 1.0 / max(v, 1e-6)

    def read_ranges(self):
        return {
            "fl": self.range_value(self.ds_front_left),
            "l": self.range_value(self.ds_left),
            "r": self.range_value(self.ds_right),
            "fr": self.range_value(self.ds_front_right),
            "front": avg([self.range_value(self.ds_front_left), self.range_value(self.ds_front_right), self.range_value(self.ds_front)]),
            "back": self.range_value(self.ds_back),
        }

    def read_lidar_sector(self, center_deg, width_deg=16):
        if self.lidar is None:
            return None
        try:
            rng = self.lidar.getRangeImage()
            if not rng:
                return None
            n = len(rng)
            center = int((center_deg % 360) / 360.0 * n)
            half = max(1, int(width_deg / 360.0 * n / 2))
            vals = []
            for k in range(center - half, center + half + 1):
                idx = k % n
                val = rng[idx]
                if math.isfinite(val):
                    vals.append(val)
            if not vals:
                return None
            return min(vals)
        except Exception:
            return None

    def request_game_info(self):
        t = self.robot.getTime()
        if t - self.last_game_info_req >= GAME_INFO_PERIOD:
            self.emitter.send(struct.pack('c', b'G'))
            self.last_game_info_req = t

    def handle_receiver(self):
        messages = []
        if self.receiver is None:
            return messages
        while self.receiver.getQueueLength() > 0:
            raw = self.receiver.getBytes()
            messages.append(raw)
            try:
                if len(raw) == 1:
                    tag = struct.unpack('c', raw)[0].decode('utf-8')
                    if tag == 'L':
                        pass
                elif len(raw) == 16:
                    tup = struct.unpack('c f i i', raw)
                    if tup[0].decode('utf-8') == 'G':
                        self.score = float(tup[1])
                        self.time_left_sim = int(tup[2])
                        self.time_left_real = int(tup[3])
            except Exception:
                pass
            self.receiver.nextPacket()
        return messages

    def report_token(self, token_type):
        pos = self.gps.getValues()
        x_cm = int(round(pos[0] * 100.0))
        z_cm = int(round(pos[2] * 100.0))
        self.emitter.send(struct.pack('i i c', x_cm, z_cm, token_type.encode('utf-8')))

    def call_lop(self):
        self.emitter.send(struct.pack('c', b'L'))

    def send_exit(self):
        self.emitter.send(struct.pack('c', b'E'))

    def send_map(self, matrix):
        if np is None:
            return
        s = matrix.shape
        s_bytes = struct.pack('2i', *s)
        flat = ','.join(matrix.flatten()).encode('utf-8')
        self.emitter.send(s_bytes + flat)
        self.emitter.send(struct.pack('c', b'M'))


class FloorClassifier:
    def __init__(self, io: RobotIO):
        self.io = io

    def _mean_rgb(self, cam):
        if cam is None or np is None:
            return None
        try:
            raw = cam.getImage()
            h, w = cam.getHeight(), cam.getWidth()
            img = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
            rgb = img[:, :, :3].astype(np.float32)
            return tuple(np.mean(rgb.reshape(-1, 3), axis=0))
        except Exception:
            return None

    def classify(self):
        rgb = self._mean_rgb(self.io.color)
        if rgb is None:
            return None
        r, g, b = rgb
        mx = max(r, g, b)
        mn = min(r, g, b)

        # Black hole / black border
        if mx < 45:
            return '2'

        # Silver checkpoint
        if mx > 150 and (mx - mn) < 30:
            return '4'

        # Brown swamp
        if r > 90 and 50 < g < 140 and b < 90 and r > g > b:
            return '3'

        # Strong colors for passages / start tile
        if g > 120 and r < 110 and b < 110:
            # Start tile is green too, but passages 1<->4 are also green.
            # We treat green as start only in area 1 before any passage has been traversed.
            return 'green'
        if b > 120 and r < 100 and g < 130:
            return 'blue'
        if r > 150 and g > 150 and b < 90:
            return 'yellow'
        if r > 120 and b > 120 and g < 120:
            return 'purple'
        if r > 170 and 80 < g < 170 and b < 90:
            return 'orange'
        if r > 150 and g < 100 and b < 100:
            return 'red'

        return '0'


class VictimDetector:
    """
    Conservative token detector.
    - Cognitive targets: higher confidence.
    - Letter victims: heuristic only; reports only when confidence is strong.
    - Fake 3D victims are NOT reliably rejected with this robot configuration; therefore
      the detector deliberately requires high 2D confidence and close range.
    """
    def __init__(self, io: RobotIO):
        self.io = io
        self.last_report_time = defaultdict(lambda: -999.0)

    def _cam_img(self, cam):
        if cam is None or np is None:
            return None
        try:
            raw = cam.getImage()
            h, w = cam.getHeight(), cam.getWidth()
            img = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
            return img[:, :, :3].copy()
        except Exception:
            return None

    def _color_to_value(self, bgr):
        b, g, r = [int(x) for x in bgr]
        if r < 60 and g < 60 and b < 60:
            return -2, 'black'
        if r > 140 and g < 100 and b < 100:
            return -1, 'red'
        if r > 140 and g > 140 and b < 120:
            return 0, 'yellow'
        if g > 120 and r < 130 and b < 130:
            return 1, 'green'
        if b > 120 and r < 120 and g < 140:
            return 2, 'blue'
        return None, 'unknown'

    def _detect_cognitive(self, img):
        if img is None or cv2 is None or np is None:
            return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                                   param1=80, param2=10, minRadius=4, maxRadius=18)
        if circles is None:
            return None
        circles = np.round(circles[0]).astype(int)
        circles = sorted(circles, key=lambda c: c[2], reverse=True)
        x, y, r = circles[0]
        if r < 5:
            return None

        samples = []
        ring_fracs = [0.10, 0.28, 0.46, 0.64, 0.82]
        for frac in ring_fracs:
            px = int(round(x + frac * r))
            py = int(round(y))
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                samples.append(img[py, px])
            else:
                return None

        total = 0
        names = []
        for s in samples:
            val, name = self._color_to_value(s)
            if val is None:
                return None
            total += val
            names.append(name)

        token = {0: 'F', 1: 'P', 2: 'C', 3: 'O'}.get(total)
        if token is None:
            return None
        return {
            'kind': 'cognitive',
            'token': token,
            'confidence': 0.90,
            'debug': {'rings': names, 'sum': total}
        }

    def _detect_letter(self, img):
        if img is None or cv2 is None or np is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        thr = cv2.medianBlur(thr, 3)
        contours, hierarchy = cv2.findContours(thr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find main contour close to center.
        h, w = gray.shape
        cx_img, cy_img = w / 2, h / 2
        best = None
        best_score = -1
        for c in contours:
            a = cv2.contourArea(c)
            if a < 40:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            cx = x + ww / 2
            cy = y + hh / 2
            score = a - 2.0 * math.hypot(cx - cx_img, cy - cy_img)
            if score > best_score:
                best_score = score
                best = (c, x, y, ww, hh, a)
        if best is None:
            return None

        _, x, y, ww, hh, a = best
        roi = thr[max(0, y-1):min(h, y+hh+1), max(0, x-1):min(w, x+ww+1)]
        if roi.size == 0:
            return None
        fill = float(np.count_nonzero(roi)) / float(roi.size)
        aspect = ww / max(hh, 1)

        # Simple structure cues.
        # Omega: open-top bowl shape -> low fill, broad aspect, no central hole.
        # Phi: central closed loop + vertical stroke -> medium fill, aspect near 1.
        # Psi: three-pronged top + stem -> taller, center-heavy vertical occupancy.
        hole_count = 0
        if hierarchy is not None:
            hier = hierarchy[0]
            for i, c in enumerate(contours):
                if hier[i][3] != -1 and cv2.contourArea(c) > 10:
                    hole_count += 1

        col_sum = np.sum(roi > 0, axis=0)
        row_sum = np.sum(roi > 0, axis=1)
        center_col = col_sum[len(col_sum)//2] / max(1, roi.shape[0])
        top_band = np.mean(row_sum[:max(1, len(row_sum)//3)]) / max(1, roi.shape[1])
        bottom_band = np.mean(row_sum[-max(1, len(row_sum)//3):]) / max(1, roi.shape[1])

        label = None
        confidence = 0.0

        if hole_count >= 1 and 0.55 < aspect < 1.35 and 0.15 < fill < 0.65 and center_col > 0.35:
            label, confidence = 'H', 0.72   # Φ -> harmed victim code H
        elif aspect > 0.85 and top_band > bottom_band * 1.2 and center_col > 0.30:
            label, confidence = 'S', 0.68   # Ψ -> stable victim code S
        elif 0.20 < fill < 0.55 and bottom_band > top_band * 0.9 and center_col < 0.55:
            label, confidence = 'U', 0.66   # Ω -> unharmed victim code U

        if label is None:
            return None
        return {
            'kind': 'letter',
            'token': label,
            'confidence': confidence,
            'debug': {'fill': fill, 'aspect': aspect, 'holes': hole_count}
        }

    def scan_and_maybe_report(self, direction_label, wall_distance, force_close=False):
        cam = self.io.camera_left if direction_label == 'left' else self.io.camera_right
        img = self._cam_img(cam)
        if img is None:
            return None

        # Only report when we are fairly close to a wall.
        if wall_distance is not None and wall_distance > 0.16 and not force_close:
            return None

        token = self._detect_cognitive(img)
        if token is None:
            token = self._detect_letter(img)
        if token is None:
            return None

        tnow = self.io.robot.getTime()
        key = f"{self.io.pose.snap_grid()}:{direction_label}:{token['token']}"
        if tnow - self.last_report_time[key] < REPORT_COOLDOWN:
            return None

        # Conservative thresholds; deliberately avoid low-confidence fake reports.
        min_conf = 0.82 if token['kind'] == 'cognitive' else 0.75
        if token['confidence'] < min_conf:
            return None

        self.io.stop()
        self._delay_seconds(VICTIM_STOP_TIME)
        self.io.report_token(token['token'])
        self.last_report_time[key] = self.io.robot.getTime()
        return token

    def _delay_seconds(self, sec):
        start = self.io.robot.getTime()
        while self.io.robot.step(TIME_STEP) != -1:
            self.io.update_pose()
            self.io.handle_receiver()
            if self.io.robot.getTime() - start >= sec:
                return


class GridMap:
    def __init__(self):
        self.cells = defaultdict(self._new_cell)
        self.start = None
        self.last_checkpoint = None
        self.reported = set()
        self.visited_order = []
        self.entered_area4 = False

    def _new_cell(self):
        return {
            'type': '0',
            'visited': 0,
            'area': None,
            'walls': {'N': None, 'E': None, 'S': None, 'W': None},
        }

    def mark_type(self, node, typ):
        cell = self.cells[node]
        if typ is None:
            return
        if typ == 'green' and self.start is None:
            cell['type'] = '5'
            self.start = node
            return
        if typ in PASSAGE_CODES:
            cell['type'] = PASSAGE_CODES[typ]
            return
        if typ in {'0', '2', '3', '4', '5', 'x'}:
            cell['type'] = typ

    def touch(self, node):
        cell = self.cells[node]
        cell['visited'] += 1
        if node not in self.visited_order:
            self.visited_order.append(node)

    def ensure_start(self, node):
        if self.start is None:
            self.start = node
            self.cells[node]['type'] = '5'
        elif self.cells[self.start]['type'] != '5':
            self.cells[self.start]['type'] = '5'

    def set_wall(self, node, d, present):
        self.cells[node]['walls'][d] = bool(present)
        dx, dz = DIR_VEC[d]
        neigh = (node[0] + dx, node[1] + dz)
        self.cells[neigh]['walls'][OPPOSITE[d]] = bool(present)

    def open_neighbors(self, node):
        out = []
        c = self.cells[node]
        for d in DIRS:
            if c['walls'][d] is False:
                dx, dz = DIR_VEC[d]
                out.append(((node[0] + dx, node[1] + dz), d))
        return out

    def unknown_neighbors(self, node):
        out = []
        c = self.cells[node]
        for d in DIRS:
            if c['walls'][d] is False:
                dx, dz = DIR_VEC[d]
                n = (node[0] + dx, node[1] + dz)
                if self.cells[n]['visited'] == 0:
                    out.append((n, d))
        return out

    def nearest_frontier_path(self, start):
        q = deque([start])
        prev = {start: None}
        goal = None
        while q:
            cur = q.popleft()
            if self.cells[cur]['visited'] == 0:
                goal = cur
                break
            for nxt, _ in self.open_neighbors(cur):
                if nxt in prev:
                    continue
                # avoid known holes
                if self.cells[nxt]['type'] == '2':
                    continue
                prev[nxt] = cur
                q.append(nxt)
        if goal is None:
            return []
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return list(reversed(path))

    def path_to(self, start, goal):
        q = deque([start])
        prev = {start: None}
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for nxt, _ in self.open_neighbors(cur):
                if nxt in prev:
                    continue
                if self.cells[nxt]['type'] == '2':
                    continue
                prev[nxt] = cur
                q.append(nxt)
        if goal not in prev:
            return []
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return list(reversed(path))

    def to_matrix(self):
        if not self.cells:
            return None
        xs = [x for x, _ in self.cells.keys()]
        zs = [z for _, z in self.cells.keys()]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        w = max_x - min_x + 1
        h = max_z - min_z + 1
        if np is None:
            return None
        m = np.full((h, w), '0', dtype='<U8')
        for (x, z), c in self.cells.items():
            m[z - min_z, x - min_x] = c['type']
        return m


class Navigator:
    def __init__(self, io: RobotIO, fmap: GridMap, floor: FloorClassifier):
        self.io = io
        self.map = fmap
        self.floor = floor
        self.last_heading_error = 0.0
        self.command_start = self.io.robot.getTime()
        self.last_progress_pos = self.io.pose.pos()
        self.last_progress_time = self.io.robot.getTime()
        self.local_recoveries = 0

    def update_local_map(self):
        node = self.io.pose.snap_grid()
        self.map.touch(node)
        self.map.ensure_start(node)
        tile_type = self.floor.classify()
        self.map.mark_type(node, tile_type)
        if self.map.cells[node]['type'] == '4':
            self.map.last_checkpoint = node

        yaw = self.io.pose.heading()
        facing = self._quantize_heading(yaw)
        left = LEFT_OF[facing]
        right = RIGHT_OF[facing]
        back = OPPOSITE[facing]
        r = self.io.read_ranges()

        # Fuse lidar sectors if available.
        front_est = avg([r['front'], self.io.read_lidar_sector(math.degrees(yaw), 20)], default=r['front'])
        left_est = avg([r['l'], self.io.read_lidar_sector(math.degrees(yaw - math.pi/2), 20)], default=r['l'])
        right_est = avg([r['r'], self.io.read_lidar_sector(math.degrees(yaw + math.pi/2), 20)], default=r['r'])
        back_est = avg([r['back'], self.io.read_lidar_sector(math.degrees(yaw + math.pi), 20)], default=r['back'])

        self.map.set_wall(node, facing, front_est is not None and front_est < WALL_NEAR)
        if left_est is not None:
            self.map.set_wall(node, left, left_est < WALL_NEAR)
        if right_est is not None:
            self.map.set_wall(node, right, right_est < WALL_NEAR)
        if back_est is not None:
            self.map.set_wall(node, back, back_est < WALL_NEAR)

    def _quantize_heading(self, yaw):
        vals = [(abs(wrap_to_pi(yaw - DIR_TO_YAW[d])), d) for d in DIRS]
        vals.sort(key=lambda t: t[0])
        return vals[0][1]

    def drive_to_node(self, node):
        tx = node[0] * CELL
        tz = node[1] * CELL
        return self.drive_to_point(tx, tz)

    def drive_to_point(self, tx, tz):
        while self.io.robot.step(TIME_STEP) != -1:
            self.io.update_pose()
            msgs = self.io.handle_receiver()
            self.io.request_game_info()

            for raw in msgs:
                if len(raw) == 1:
                    try:
                        if struct.unpack('c', raw)[0].decode('utf-8') == 'L':
                            self.io.stop()
                            return 'lop'
                    except Exception:
                        pass

            x, z = self.io.pose.pos()
            yaw = self.io.pose.heading()
            dx = tx - x
            dz = tz - z
            dist = math.hypot(dx, dz)
            if dist < POSITION_TOL:
                self.io.stop()
                return 'ok'

            target_yaw = math.atan2(dz, dx)
            err = wrap_to_pi(target_yaw - yaw)
            derr = (err - self.last_heading_error) / (TIME_STEP / 1000.0)
            self.last_heading_error = err

            v = clamp(LINEAR_KP * dist, 0.10 * MAX_SPEED, MAX_CRUISE)
            if abs(err) > math.radians(30):
                v *= 0.45
            w = clamp(ANGULAR_KP * err + ANGULAR_KD * derr, -MAX_TURN, MAX_TURN)
            self.io.set_speed(v - w, v + w)

            # progress monitor
            if math.hypot(x - self.last_progress_pos[0], z - self.last_progress_pos[1]) > STUCK_DIST:
                self.last_progress_pos = (x, z)
                self.last_progress_time = self.io.robot.getTime()
                self.local_recoveries = 0
            elif self.io.robot.getTime() - self.last_progress_time > STUCK_TIME:
                self.io.stop()
                if self.local_recoveries < MAX_LOCAL_RECOVERY:
                    self.local_recoveries += 1
                    self.local_recovery()
                    self.last_progress_time = self.io.robot.getTime()
                else:
                    self.io.call_lop()
                    return 'requested_lop'

        return 'done'

    def rotate_to_dir(self, d):
        return self.rotate_to_yaw(DIR_TO_YAW[d])

    def rotate_to_yaw(self, target):
        while self.io.robot.step(TIME_STEP) != -1:
            self.io.update_pose()
            self.io.handle_receiver()
            err = wrap_to_pi(target - self.io.pose.heading())
            if abs(err) < HEADING_TOL:
                self.io.stop()
                return 'ok'
            w = clamp(ANGULAR_KP * err, -MAX_TURN, MAX_TURN)
            self.io.set_speed(-w, w)
        return 'done'

    def classify_intersection(self):
        node = self.io.pose.snap_grid()
        open_dirs = []
        for d in DIRS:
            if self.map.cells[node]['walls'][d] is False:
                open_dirs.append(d)
        if len(open_dirs) == 0:
            return 'blocked'
        if len(open_dirs) == 1:
            return 'dead_end'
        if len(open_dirs) == 2:
            a, b = open_dirs
            if OPPOSITE[a] == b:
                return 'corridor'
            return 'corner'
        if len(open_dirs) == 3:
            return 'tee'
        return 'cross'

    def local_recovery(self):
        # Simple but effective: back off, then oscillatory turn.
        start = self.io.robot.getTime()
        while self.io.robot.step(TIME_STEP) != -1:
            self.io.update_pose()
            self.io.handle_receiver()
            if self.io.robot.getTime() - start > 0.45:
                break
            self.io.set_speed(-0.35 * MAX_SPEED, -0.35 * MAX_SPEED)

        start = self.io.robot.getTime()
        left_first = (self.local_recoveries % 2 == 1)
        while self.io.robot.step(TIME_STEP) != -1:
            self.io.update_pose()
            self.io.handle_receiver()
            if self.io.robot.getTime() - start > 0.55:
                break
            turn = 0.38 * MAX_SPEED
            if left_first:
                self.io.set_speed(-turn, turn)
            else:
                self.io.set_speed(turn, -turn)
        self.io.stop()


class RescueController:
    def __init__(self):
        self.robot = Robot()
        self.io = RobotIO(self.robot)
        self.floor = FloorClassifier(self.io)
        self.detector = VictimDetector(self.io)
        self.map = GridMap()
        self.nav = Navigator(self.io, self.map, self.floor)
        self.mode = 'explore'
        self.current_path = []
        self.last_map_submit = -999
        self.reported_any = False

    def node_to_dir(self, cur, nxt):
        dx = nxt[0] - cur[0]
        dz = nxt[1] - cur[1]
        for d, v in DIR_VEC.items():
            if v == (dx, dz):
                return d
        return None

    def choose_path(self, current):
        # Prefer immediately reachable unknown neighbors.
        unknowns = self.map.unknown_neighbors(current)
        if unknowns:
            return [current, unknowns[0][0]]

        # Otherwise BFS to nearest frontier.
        path = self.map.nearest_frontier_path(current)
        if path:
            return path

        # Exploration exhausted: return to start if known.
        if self.map.start is not None and current != self.map.start:
            return self.map.path_to(current, self.map.start)
        return []

    def maybe_report_side_tokens(self):
        ranges = self.io.read_ranges()
        left_token = self.detector.scan_and_maybe_report('left', ranges['l'])
        right_token = self.detector.scan_and_maybe_report('right', ranges['r'])
        if left_token or right_token:
            self.reported_any = True
            return True
        return False

    def maybe_submit_map(self):
        t = self.io.robot.getTime()
        if t - self.last_map_submit < 10.0:
            return
        m = self.map.to_matrix()
        if m is not None:
            self.io.send_map(m)
            self.last_map_submit = t

    def should_exit(self, current):
        if self.map.start is None:
            return False
        low_time = (self.io.time_left_sim <= EXIT_TIME_LEFT_SIM or self.io.time_left_real <= EXIT_TIME_LEFT_REAL)
        exploration_done = not self.map.nearest_frontier_path(current)
        return self.reported_any and current == self.map.start and (low_time or exploration_done)

    def handle_area4_fallback(self):
        # Conservative continuous-space fallback: wall-follow with side preference
        # until an exit condition or a new structured tile is encountered.
        start = self.io.robot.getTime()
        while self.io.robot.step(TIME_STEP) != -1:
            self.io.update_pose()
            msgs = self.io.handle_receiver()
            self.io.request_game_info()
            for raw in msgs:
                if len(raw) == 1:
                    try:
                        if struct.unpack('c', raw)[0].decode('utf-8') == 'L':
                            return
                    except Exception:
                        pass

            self.maybe_report_side_tokens()
            ranges = self.io.read_ranges()
            front = ranges['front'] if ranges['front'] is not None else 1.0
            left = ranges['l'] if ranges['l'] is not None else 1.0
            right = ranges['r'] if ranges['r'] is not None else 1.0

            if front < WALL_VERY_NEAR:
                self.io.set_speed(-0.25 * MAX_SPEED, 0.25 * MAX_SPEED)
            else:
                target = 0.10
                err = target - left
                corr = clamp(4.0 * err, -0.25 * MAX_SPEED, 0.25 * MAX_SPEED)
                base = 0.35 * MAX_SPEED
                self.io.set_speed(base + corr, base - corr)

            typ = self.floor.classify()
            if typ in {'0', '2', '3', '4'} or typ in PASSAGE_CODES or typ == 'green':
                # Structured area again; break back to planner.
                return
            if self.io.robot.getTime() - start > 12.0:
                return

    def main(self):
        while self.io.step() != -1:
            self.io.update_pose()
            msgs = self.io.handle_receiver()
            self.io.request_game_info()

            for raw in msgs:
                if len(raw) == 1:
                    try:
                        if struct.unpack('c', raw)[0].decode('utf-8') == 'L':
                            # The engine already reset the robot; clear transient plan.
                            self.current_path = []
                    except Exception:
                        pass

            self.nav.update_local_map()
            current = self.io.pose.snap_grid()
            self.maybe_report_side_tokens()
            self.maybe_submit_map()

            # Area 4 fallback trigger: if no grid path exists and time remains.
            if self.map.cells[current]['type'] == '0' and self.map.start is not None and not self.current_path:
                pass

            if self.should_exit(current):
                self.io.stop()
                self.maybe_submit_map()
                self.io.send_exit()
                break

            if not self.current_path or self.current_path[-1] == current:
                self.current_path = self.choose_path(current)
                if len(self.current_path) <= 1:
                    # Nothing else to do; try exit if at start, else pause then continue.
                    if self.map.start is not None and current == self.map.start and self.reported_any:
                        self.maybe_submit_map()
                        self.io.send_exit()
                        break
                    self.handle_area4_fallback()
                    continue

            # Next node on path.
            if self.current_path[0] == current:
                next_node = self.current_path[1]
            else:
                # resync path
                self.current_path = self.choose_path(current)
                if len(self.current_path) <= 1:
                    continue
                next_node = self.current_path[1]

            d = self.node_to_dir(current, next_node)
            if d is None:
                self.current_path = []
                continue

            # If known blocked, invalidate plan.
            if self.map.cells[current]['walls'][d] is True:
                self.current_path = []
                continue

            rot_status = self.nav.rotate_to_dir(d)
            if rot_status == 'lop':
                self.current_path = []
                continue
            move_status = self.nav.drive_to_node(next_node)
            if move_status in {'lop', 'requested_lop'}:
                self.current_path = []
                continue

            # Dead-end and backtracking naturally happen via BFS frontier selection.


if __name__ == '__main__':
    controller = RescueController()
    controller.main()
