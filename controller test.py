from controller import Robot
import math
import time

TIME_STEP = 32
MAX_SPEED = 6.28
TILE_SIZE_M = 0.12
PROGRESS_TIMEOUT = 4.0


class RescueMazeController:

    def __init__(self):
        self.robot = Robot()
        self.start_time = self.robot.getTime()

        # =========================
        # 🔧 FIXED PART (motors)
        # =========================
        self.left_motor = self._get_device(["wheel1 motor"], required=True)
        self.right_motor = self._get_device(["wheel2 motor"], required=True)
        # =========================

        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

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

        if self.lidar is not None:
            try:
                self.lidar.enablePointCloud()
            except Exception:
                pass

        self.last_progress_time = self.robot.getTime()
        self.last_progress_pos = self.get_position_xy()
        self.last_turn_dir = 1
        self.stuck_recovery_until = 0.0
        self.spin_until = 0.0
        self.spin_dir = 1

        self.visited = set()
        self.special_tiles = {}
        self.start_cell = None
        self.last_color_name = "unknown"

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
            except:
                pass
        return dev

    # =========================
    # Sensors
    # =========================
    def get_position_xy(self):
        if not self.gps:
            return (0, 0)
        vals = self.gps.getValues()
        return vals[0], vals[2]

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
        return min(self.dist(self.ds_left), self.dist(self.ds_front_left))

    def read_right(self):
        return min(self.dist(self.ds_right), self.dist(self.ds_front_right))

    # =========================
    # Movement
    # =========================
    def set_speed(self, l, r):
        self.left_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, l)))
        self.right_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, r)))

    def forward(self):
        self.set_speed(0.5 * MAX_SPEED, 0.5 * MAX_SPEED)

    def turn_left(self):
        self.set_speed(-0.4 * MAX_SPEED, 0.4 * MAX_SPEED)

    def turn_right(self):
        self.set_speed(0.4 * MAX_SPEED, -0.4 * MAX_SPEED)

    # =========================
    # Behavior
    # =========================
    def wall_follow(self):
        front = self.read_front()
        right = self.read_right()

        if front < 0.12:
            self.turn_right()
            return

        error = 0.2 - right
        correction = 3 * error

        left_speed = 0.5 * MAX_SPEED - correction
        right_speed = 0.5 * MAX_SPEED + correction

        self.set_speed(left_speed, right_speed)

    # =========================
    # Main loop
    # =========================
    def run(self):
        while self.robot.step(TIME_STEP) != -1:
            self.wall_follow()


if __name__ == "__main__":
    controller = RescueMazeController()
    controller.run()