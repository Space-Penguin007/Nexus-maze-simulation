from controller import Robot
import math
import numpy as np
import cv2
from collections import deque

TIME_STEP = 32
MAX_SPEED = 6.28
TILE_SIZE_M = 0.12
PROGRESS_TIMEOUT = 2.5

BLACK_TURN_TIME = 0.45       # seconds
SWAMP_BACK_TIME = 0.35       # seconds
SWAMP_TURN_TIME = 0.60       # seconds

# Anti-stuck tuning
MIN_PROGRESS_DIST = 0.012    # meters; if below this for too long, consider stuck
RECOVERY_BACK_TIME = 0.45    # seconds
RECOVERY_TURN_TIME = 0.75    # seconds
RECOVERY_FORWARD_TIME = 0.20 # seconds
BACK_CLEARANCE = 0.08        # meters; minimum rear clearance to back up

# PID wall-follow tuning (stabilized)
TARGET_WALL_DISTANCE = 0.28  # desired distance from right wall (meters)
WALL_LOST_DISTANCE = 0.35     # if farther than this, treat wall as lost
PID_KP = 2.0
PID_KI = 0.05
PID_KD = 0.20
PID_INTEGRAL_CLAMP = 0.18
BASE_FORWARD_SPEED = 0.42 * MAX_SPEED
SEARCH_WALL_LEFT_SPEED = 0.46 * MAX_SPEED
SEARCH_WALL_RIGHT_SPEED = 0.34 * MAX_SPEED
DERIVATIVE_ALPHA = 0.75       # 0..1; higher = smoother derivative
MAX_PID_CORRECTION = 0.70 * MAX_SPEED

MIN_CONTOUR_AREA = 20
MAX_CONTOUR_AREA_RATIO = 0.60
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.8

TEMPLATE_SIZE = 64
MATCH_THRESHOLD = 0.48
CONFIRM_FRAMES_REQUIRED = 3
CONFIRM_WINDOW = 5

DUPLICATE_DISTANCE_M = 0.12
SIDE_LEFT = "left"
SIDE_RIGHT = "right"

MIN_MARGIN_FOR_HISTORY = 0.018
MIN_MARGIN_FOR_CONFIRM = 0.028
CONFIRM_TAIL_FRAMES = 4
CONFIRM_SUPPORT_REQUIRED = 3
SIDE_WALL_MAX_DIST = 0.24

CROP_X0 = 0.20
CROP_X1 = 0.95
CROP_Y0 = 0.15
CROP_Y1 = 0.85


CHECKPOINT_MIN_SEPARATION_M = 0.08
START_ZONE_RADIUS_M = 0.06


class VictimRecognizer:
    def __init__(self, robot, gps, camera_left, camera_right):
        self.robot = robot
        self.gps = gps
        self.camera_left = camera_left
        self.camera_right = camera_right

        self.templates = self._build_templates()
        self.left_history = deque(maxlen=CONFIRM_WINDOW)
        self.right_history = deque(maxlen=CONFIRM_WINDOW)

        # Store already reported victims:
        # list of dicts: {"x":..., "y":..., "label":..., "side":...}
        self.reported = []

    # =========================================================
    # Public API
    # =========================================================
    def update(self, left_wall_dist=None, right_wall_dist=None):
        left_result = self._process_camera(
            self.camera_left, SIDE_LEFT, self.left_history, left_wall_dist
        )
        right_result = self._process_camera(
            self.camera_right, SIDE_RIGHT, self.right_history, right_wall_dist
        )

        confirmed_events = []

        if left_result is not None:
            confirmed_events.append(left_result)

        if right_result is not None:
            confirmed_events.append(right_result)

        return confirmed_events

    # =========================================================
    # Core processing
    # =========================================================
    def _process_camera(self, camera, side, history, wall_dist=None):
        if camera is None:
            return None

        if wall_dist is not None and wall_dist > SIDE_WALL_MAX_DIST:
            history.append(None)
            return None

        frame = self._camera_to_bgr(camera)
        if frame is None:
            history.append(None)
            return None

        roi = self._crop_wall_region(frame)
        binary = self._preprocess(roi)
        candidate = self._extract_candidate(binary)

        if candidate is None:
            history.append(None)
            return None

        result = self._classify_candidate(candidate)

        label = result["best_label"]
        score = result["best_dist"]
        margin = result["margin"]
        norm_patch = result["patch"]

        if label is None or score > MATCH_THRESHOLD or margin < MIN_MARGIN_FOR_HISTORY:
            history.append(None)
            return None

        entry = {
            "label": label,
            "score": score,
            "margin": margin,
        }
        history.append(entry)

        if self._is_confirmed(history, label):
            pos = self._get_position_xy()
            now = self.robot.getTime()

            if not self._is_duplicate(pos[0], pos[1], side):
                self._store_report(pos[0], pos[1], label, side, now)
                return {
                    "label": label,
                    "score": score,
                    "side": side,
                    "x": pos[0],
                    "y": pos[1],
                    "patch": norm_patch,
                }

        return None

    # =========================================================
    # Camera conversion
    # =========================================================
    def _camera_to_bgr(self, camera):
        try:
            width = camera.getWidth()
            height = camera.getHeight()
            image = camera.getImage()

            if image is None:
                return None

            # Webots image is BGRA-like packed bytes for Camera
            arr = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
            bgr = arr[:, :, :3].copy()
            return bgr
        except Exception as e:
            print(f"[VICTIM] camera conversion failed: {e}")
            return None

    # =========================================================
    # ROI / preprocessing
    # =========================================================
    def _crop_wall_region(self, frame):
        h, w = frame.shape[:2]
        x0 = int(w * CROP_X0)
        x1 = int(w * CROP_X1)
        y0 = int(h * CROP_Y0)
        y1 = int(h * CROP_Y1)
        return frame[y0:y1, x0:x1]

    def _preprocess(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Blur slightly to reduce pixel noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive threshold often works well in simulation
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 7
        )

        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    # =========================================================
    # Candidate extraction
    # =========================================================
    def _extract_candidate(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        img_h, img_w = binary.shape[:2]
        img_area = img_h * img_w

        best = None
        best_score = -1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            if area > MAX_CONTOUR_AREA_RATIO * img_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue

            aspect = w / float(h)
            if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
                continue

            fill_ratio = area / float(w * h + 1e-6)

            # Prefer reasonably filled, centered, large-ish candidates
            cx = x + w / 2.0
            cy = y + h / 2.0
            center_dist = math.hypot(cx - img_w / 2.0, cy - img_h / 2.0)

            score = area * (0.8 + fill_ratio) - 0.8 * center_dist

            if score > best_score:
                best_score = score
                best = (x, y, w, h)

        if best is None:
            return None

        x, y, w, h = best

        # Add margin around candidate
        pad = 3
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(binary.shape[1], x + w + pad)
        y1 = min(binary.shape[0], y + h + pad)

        patch = binary[y0:y1, x0:x1]
        if patch.size == 0:
            return None

        return patch

    # =========================================================
    # Classification
    # =========================================================
    def _classify_candidate(self, patch):
        norm = self._normalize_patch(patch)

        results = []

        for rotated in (
            norm,
            np.rot90(norm, 1).copy(),
            np.rot90(norm, 2).copy(),
            np.rot90(norm, 3).copy(),
        ):
            for label, templ in self.templates.items():
                dist = self._template_distance(rotated, templ)
                results.append((dist, label, rotated))

        results.sort(key=lambda x: x[0])

        best_dist, best_label, best_patch = results[0]
        second_dist = results[1][0] if len(results) > 1 else 999.0
        margin = second_dist - best_dist

        return {
            "best_label": best_label,
            "best_dist": best_dist,
            "second_dist": second_dist,
            "margin": margin,
            "patch": best_patch,
        }

    def _normalize_patch(self, patch):
        # Resize while preserving aspect ratio, place centered on square canvas
        h, w = patch.shape[:2]
        canvas = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.uint8)

        scale = min((TEMPLATE_SIZE - 8) / max(w, 1), (TEMPLATE_SIZE - 8) / max(h, 1))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        x0 = (TEMPLATE_SIZE - new_w) // 2
        y0 = (TEMPLATE_SIZE - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

        # Normalize to 0/1 float image
        out = (canvas > 127).astype(np.float32)
        return out

    def _template_distance(self, img, templ):
        # Mean absolute difference
        return float(np.mean(np.abs(img - templ)))

    # =========================================================
    # Temporal confirmation
    # =========================================================
    def _is_confirmed(self, history, label):
        tail = list(history)[-CONFIRM_TAIL_FRAMES:]
        if len(tail) < CONFIRM_TAIL_FRAMES:
            return False

        strong_matches = [
            h for h in tail
            if h is not None
            and h["label"] == label
            and h["margin"] >= MIN_MARGIN_FOR_CONFIRM
            and h["score"] <= MATCH_THRESHOLD
        ]

        if len(strong_matches) < CONFIRM_SUPPORT_REQUIRED:
            return False

        last2 = tail[-2:]
        for h in last2:
            if h is None:
                return False
            if h["label"] != label:
                return False
            if h["margin"] < MIN_MARGIN_FOR_CONFIRM:
                return False
            if h["score"] > MATCH_THRESHOLD:
                return False

        return True

    # =========================================================
    # Duplicate suppression
    # =========================================================
    def _get_position_xy(self):
        if self.gps is None:
            return (0.0, 0.0)
        vals = self.gps.getValues()
        return (vals[0], vals[2])

    def _is_duplicate(self, x, y, side):
        for item in self.reported:
            if item["side"] != side:
                continue

            d = math.hypot(x - item["x"], y - item["y"])
            if d < DUPLICATE_DISTANCE_M:
                return True

        return False


    def _store_report(self, x, y, label, side, now):
        self.reported.append({
            "x": x,
            "y": y,
            "label": label,
            "side": side,
            "time": now,
        })
            

    # =========================================================
    # Template generation
    # =========================================================
    def _build_templates(self):
        return {
            "H": self._make_letter_template("H"),
            "S": self._make_letter_template("S"),
            "U": self._make_letter_template("U"),
        }

    def _make_letter_template(self, letter):
        img = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.8
        thickness = 5

        (tw, th), baseline = cv2.getTextSize(letter, font, scale, thickness)
        x = (TEMPLATE_SIZE - tw) // 2
        y = (TEMPLATE_SIZE + th) // 2

        cv2.putText(img, letter, (x, y), font, scale, 255, thickness, cv2.LINE_AA)

        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return (img > 127).astype(np.float32)
    


class RescueMazeController:

    def __init__(self):
        self.robot = Robot()
        self.start_time = self.robot.getTime()

        # =========================
        # Motors
        # =========================
        self.left_motor = self._get_device(["wheel1 motor"], required=True)
        self.right_motor = self._get_device(["wheel2 motor"], required=True)

        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Remember last commanded speed, useful for stuck detection
        self.cmd_left = 0.0
        self.cmd_right = 0.0

        # --- Sensors ---
        self.ds_front_left = self._enable_sensor(["ds_front_left"])
        self.ds_left = self._enable_sensor(["ds_left"])
        self.ds_right = self._enable_sensor(["ds_right"])
        self.ds_front_right = self._enable_sensor(["ds_front_right"])
        self.ds_back = self._enable_sensor(["distance sensor5"])
        self.ds_front = self._enable_sensor(["distance sensor6"])

        self.gps = self._enable_sensor(["gps"])
        self.gyro = self._enable_sensor(["gyro"])
        self.color = self._enable_sensor(["color"])
        self.lidar = self._enable_sensor(["lidar"])

        self.camera_left = self._enable_sensor(["camera_left"])
        self.camera_right = self._enable_sensor(["camera_right"])

        self.recognizer = VictimRecognizer(
                robot=self.robot,
                gps=self.gps,
                camera_left=self.camera_left,
                camera_right=self.camera_right,
                
        )
        
        # grid / report anchor
        self.grid_anchor_initialized = False
        self.grid_anchor_x = 0.0
        self.grid_anchor_z = 0.0

        # scoring / rule bookkeeping
        self.start_pos_xy = None
        self.visited_checkpoints = set()
        self.real_tokens_reported_count = 0
        self.exit_sent = False

        if self.lidar is not None:
            try:
                self.lidar.enablePointCloud()
            except Exception:
                pass

        self.last_progress_time = self.robot.getTime()
        self.last_progress_pos = self.get_position_xy()
        self.last_turn_dir = 1
        self.last_color_name = "unknown"

        # black tile behavior
        self.black_turn_until = 0.0

        # swamp avoidance behavior
        self.swamp_back_until = 0.0
        self.swamp_turn_until = 0.0
        self.in_swamp_avoidance = False
        self.was_on_brown = False

        # anti-stuck behavior
        self.in_stuck_recovery = False
        self.recovery_back_until = 0.0
        self.recovery_turn_until = 0.0
        self.recovery_forward_until = 0.0
        self.recovery_turn_dir = 1  # 1 = right, -1 = left

        # wall-follow PID state
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        self.pid_prev_time = self.robot.getTime()
        self.pid_filtered_derivative = 0.0

        
    # =========================
    # Device helpers
    # =========================
    def _get_device(self, candidates, required=False):
        for name in candidates:
            try:
                dev = self.robot.getDevice(name)
                if dev is not None:
                    print(f"[INFO] Found device: {name}")
                    return dev
            except Exception:
                continue
        if required:
            raise RuntimeError(f"Missing device: {candidates}")
        return None

    def _enable_sensor(self, names):
        dev = self._get_device(names)
        if dev:
            try:
                dev.enable(TIME_STEP)
            except Exception:
                pass
        return dev

    # =========================
    # Sensors
    # =========================
    def get_position_xy(self):
        if not self.gps:
            return (0.0, 0.0)
        vals = self.gps.getValues()
        return vals[0], vals[2]
    
    def _pos_to_key(self, x, y, resolution=0.06):
        return (int(round(x / resolution)), int(round(y / resolution)))

    def _handle_checkpoint_visit(self, now):
        x, y = self.get_position_xy()
        key = self._pos_to_key(x, y, resolution=CHECKPOINT_MIN_SEPARATION_M)

        if key not in self.visited_checkpoints:
            self.visited_checkpoints.add(key)
            print(f"[CHECKPOINT] New checkpoint visited at ({x:.3f}, {y:.3f})")

    def _is_in_start_zone(self):
        if self.start_pos_xy is None:
            return False

        x, y = self.get_position_xy()
        d = math.hypot(x - self.start_pos_xy[0], y - self.start_pos_xy[1])
        return d <= START_ZONE_RADIUS_M

    def dist(self, s):
        if not s:
            return 1.0
        return s.getValue()

    def read_front(self):
        return min(
            self.dist(self.ds_front),
            self.dist(self.ds_front_left),
            self.dist(self.ds_front_right),
        )

    def read_left(self):
        return self.dist(self.ds_left)

    def read_right(self):
        # IMPORTANT: use only the true side sensor for wall-follow PID.
        # Do not mix the front-right sensor into the lateral distance loop.
        return self.dist(self.ds_right)

    def read_back(self):
        return self.dist(self.ds_back)

    # =========================
    # Color detection
    # =========================
    def read_floor_rgb(self):
        if not self.color:
            return None

        try:
            image = self.color.getImage()
            width = self.color.getWidth()
            height = self.color.getHeight()

            if image is None or width <= 0 or height <= 0:
                return None

            x = width // 2
            y = height // 2

            r = self.color.imageGetRed(image, width, x, y)
            g = self.color.imageGetGreen(image, width, x, y)
            b = self.color.imageGetBlue(image, width, x, y)

            return (r, g, b)
        except Exception as e:
            print(f"[WARN] Color read failed: {e}")
            return None

    def detect_color_name(self):
        rgb = self.read_floor_rgb()
        if rgb is None:
            return "unknown"

        r, g, b = rgb
        print(f"[COLOR] RGB = ({r}, {g}, {b})")

        # black hole edge
        if r < 60 and g < 60 and b < 60:
            return "black"

        # silver checkpoint: medium/high gray, low channel spread
        mx = max(r, g, b)
        mn = min(r, g, b)
        if 140 <= r <= 230 and 140 <= g <= 230 and 140 <= b <= 230 and (mx - mn) <= 18:
            return "silver"

        # white normal floor
        if r > 230 and g > 230 and b > 230:
            return "white"

        # brown swamp (2026 rules)
        # intentionally broad; tune later from logs
        if 70 <= r <= 170 and 40 <= g <= 120 and b <= 90 and r > g:
            return "brown"

        # passage colors / other colors
        if r > 180 and g < 120 and b < 120:
            return "red"
        if g > 180 and r < 140 and b < 140:
            return "green"
        if b > 180 and r < 140 and g < 140:
            return "blue"
        if r > 170 and g > 170 and b < 140:
            return "yellow"
        if r > 120 and b > 160 and g < 150:
            return "purple"

        return "unknown"

    # =========================
    # Movement
    # =========================
    def set_speed(self, l, r):
        l = max(-MAX_SPEED, min(MAX_SPEED, l))
        r = max(-MAX_SPEED, min(MAX_SPEED, r))
        self.cmd_left = l
        self.cmd_right = r
        self.left_motor.setVelocity(l)
        self.right_motor.setVelocity(r)

    def forward(self):
        self.set_speed(0.50 * MAX_SPEED, 0.50 * MAX_SPEED)

    def backward(self):
        self.set_speed(-0.45 * MAX_SPEED, -0.45 * MAX_SPEED)

    def turn_left(self):
        self.set_speed(-0.50 * MAX_SPEED, 0.50 * MAX_SPEED)

    def turn_right(self):
        self.set_speed(0.50 * MAX_SPEED, -0.50 * MAX_SPEED)

    def stop(self):
        self.set_speed(0.0, 0.0)

    # =========================
    # Behavior
    # =========================
    def reset_wall_pid(self, now=None):
        if now is None:
            now = self.robot.getTime()
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        self.pid_prev_time = now
        self.pid_filtered_derivative = 0.0

    def wall_follow(self, now):
        front = self.read_front()
        right = self.read_right()
        front_right = self.dist(self.ds_front_right)

        # Immediate obstacle ahead: prioritize turn.
        if front < 0.11:
            self.reset_wall_pid(now)
            self.turn_right()
            return

        # Near a right-front corner: turn a bit earlier and more decisively.
        if front_right < 0.08 and right < 0.18:
            self.reset_wall_pid(now)
            self.turn_left()
            return

        # Right wall temporarily lost: search it with a gentle right arc.
        if right > WALL_LOST_DISTANCE:
            self.reset_wall_pid(now)
            self.set_speed(SEARCH_WALL_LEFT_SPEED, SEARCH_WALL_RIGHT_SPEED)
            return

        dt = max(0.001, now - self.pid_prev_time)
        error = TARGET_WALL_DISTANCE - right

        # Integrator with anti-windup clamp.
        self.pid_integral += error * dt
        self.pid_integral = max(-PID_INTEGRAL_CLAMP, min(PID_INTEGRAL_CLAMP, self.pid_integral))

        # Derivative with simple low-pass filtering.
        raw_derivative = (error - self.pid_prev_error) / dt
        self.pid_filtered_derivative = (
            DERIVATIVE_ALPHA * self.pid_filtered_derivative
            + (1.0 - DERIVATIVE_ALPHA) * raw_derivative
        )

        correction = (
            PID_KP * error
            + PID_KI * self.pid_integral
            + PID_KD * self.pid_filtered_derivative
        )
        correction = max(-MAX_PID_CORRECTION, min(MAX_PID_CORRECTION, correction))

        left_speed = BASE_FORWARD_SPEED - correction
        right_speed = BASE_FORWARD_SPEED + correction
        self.set_speed(left_speed, right_speed)

        self.pid_prev_error = error
        self.pid_prev_time = now

    def start_swamp_avoidance(self, now):
        print("[ACTION] Orange/swamp detected -> back off and avoid")
        self.swamp_back_until = now + SWAMP_BACK_TIME
        self.swamp_turn_until = self.swamp_back_until + SWAMP_TURN_TIME
        self.in_swamp_avoidance = True
        self.reset_wall_pid(now)
        self.reset_progress_watch(now)

    def handle_swamp_avoidance(self, now):
        if now < self.swamp_back_until:
            self.backward()
            return True

        if now < self.swamp_turn_until:
            self.turn_right()
            return True

        self.in_swamp_avoidance = False
        self.reset_wall_pid(now)
        self.reset_progress_watch(now)
        return False

        # =========================
    # Anti-stuck logic
    # =========================
    def reset_progress_watch(self, now):
        self.last_progress_time = now
        self.last_progress_pos = self.get_position_xy()


    def is_trying_to_translate(self):
        # True only when both wheels are trying to move the robot
        # in the same direction with enough speed.
        avg = (abs(self.cmd_left) + abs(self.cmd_right)) * 0.5
        same_sign = self.cmd_left * self.cmd_right > 0
        return same_sign and avg > 0.20 * MAX_SPEED


    def update_progress_watch(self, now):
        pos = self.get_position_xy()
        moved = math.hypot(
            pos[0] - self.last_progress_pos[0],
            pos[1] - self.last_progress_pos[1]
        )

        # Only reset progress when actual translation happened.
        if moved >= MIN_PROGRESS_DIST:
            self.last_progress_pos = pos
            self.last_progress_time = now


    def should_start_stuck_recovery(self, now):
        if self.in_stuck_recovery:
            return False
        if self.in_swamp_avoidance:
            return False
        if now < self.black_turn_until:
            return False
        if not self.is_trying_to_translate():
            return False

        return (now - self.last_progress_time) >= PROGRESS_TIMEOUT


    def choose_recovery_turn_dir(self):
        left = self.read_left()
        right = self.read_right()

        # Turn toward the side with more free space
        if left > right + 0.02:
            self.last_turn_dir = -1   # left
            return -1

        if right > left + 0.02:
            self.last_turn_dir = 1    # right
            return 1

        # If similar, alternate
        self.last_turn_dir *= -1
        return self.last_turn_dir


    def start_stuck_recovery(self, now):
        self.in_stuck_recovery = True
        self.recovery_turn_dir = self.choose_recovery_turn_dir()

        back_clear = self.read_back()
        do_back = back_clear > BACK_CLEARANCE
        back_time = RECOVERY_BACK_TIME if do_back else 0.0

        self.recovery_back_until = now + back_time
        self.recovery_turn_until = self.recovery_back_until + RECOVERY_TURN_TIME
        self.recovery_forward_until = self.recovery_turn_until + RECOVERY_FORWARD_TIME

        direction_name = "right" if self.recovery_turn_dir == 1 else "left"
        print(f"[RECOVERY] Robot stuck -> back={do_back} rear={back_clear:.3f} turn={direction_name}")

        self.reset_wall_pid(now)
        self.reset_progress_watch(now)


    def handle_stuck_recovery(self, now):
        # Phase 1: back up only if there is enough rear clearance
        if now < self.recovery_back_until:
            if self.read_back() > 0.5 * BACK_CLEARANCE:
                self.backward()
                return True
            else:
                # Abort backing if rear becomes unsafe
                self.recovery_back_until = now
                self.recovery_turn_until = now + RECOVERY_TURN_TIME
                self.recovery_forward_until = self.recovery_turn_until + RECOVERY_FORWARD_TIME

        # Phase 2: turn until timeout
        if now < self.recovery_turn_until:
            if self.recovery_turn_dir == 1:
                self.turn_right()
            else:
                self.turn_left()
            return True

        # Phase 3: short forward burst
        if now < self.recovery_forward_until:
            self.forward()
            return True

        print("[RECOVERY] Completed")
        self.in_stuck_recovery = False
        self.reset_wall_pid(now)
        self.reset_progress_watch(now)
        return False
    # =========================
    # Main loop
    # =========================
    def run(self):
        while self.robot.step(TIME_STEP) != -1:
            now = self.robot.getTime()
            if not self.grid_anchor_initialized:
                gx, gz = self.get_position_xy()
                self.grid_anchor_x = gx
                self.grid_anchor_z = gz
                self.grid_anchor_initialized = True
                self.start_pos_xy = (gx, gz)
                print(f"[REPORT] Grid anchor set to x={gx:.3f} z={gz:.3f}")
            color_name = self.detect_color_name()
            if color_name == "silver":
                self._handle_checkpoint_visit(now)
            
            events = self.recognizer.update(self.read_left(), self.read_right())
            for ev in events:
                print(
                        f"[VICTIM CONFIRMED] label={ev['label']} "
                        f"side={ev['side']} "
                        f"pos=({ev['x']:.3f}, {ev['y']:.3f}) "
                        f"score={ev['score']:.3f}"
                    )

            if color_name != self.last_color_name:
                print(f"[INFO] Detected floor color: {color_name}")
                self.last_color_name = color_name

            # Track orange entry/exit properly to avoid repeated retriggers.
            is_on_brown = (color_name == "brown")
            brown_entry = is_on_brown and not self.was_on_brown
            self.was_on_brown = is_on_brown

            if self.in_stuck_recovery:
                if self.handle_stuck_recovery(now):
                    continue

            if self.in_swamp_avoidance:
                if self.handle_swamp_avoidance(now):
                    continue

            if brown_entry and not self.in_swamp_avoidance:
                self.start_swamp_avoidance(now)
                continue

            if color_name == "black" and now >= self.black_turn_until:
                print("[ACTION] Black tile detected -> turning right")
                self.black_turn_until = now + BLACK_TURN_TIME
                self.reset_wall_pid(now)
                self.reset_progress_watch(now)

            if now < self.black_turn_until:
                self.turn_right()
                continue

            self.wall_follow(now)
            self.update_progress_watch(now)

            if self.should_start_stuck_recovery(now):
                self.start_stuck_recovery(now)


if __name__ == "__main__":
    controller = RescueMazeController()
    controller.run()
