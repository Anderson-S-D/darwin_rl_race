import math
from controller import Supervisor


class DarwinGaitController:
    MOTOR_NAMES = [
        "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR", "ArmLowerL",
        "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR", "LegUpperL",
        "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR", "FootL",
        "Neck", "Head",
    ]

    JOINT_SIGNS = {
        "LegUpperR": -1.0,
        "LegUpperL": 1.0,
        "LegLowerR": 1.0,
        "LegLowerL": -1.0,
        "AnkleR": 1.0,
        "AnkleL": -1.0,
    }

    def __init__(self, robot: Supervisor):
        self.robot = robot
        self.motors = {}

        for name in self.MOTOR_NAMES:
            motor = self.robot.getDevice(name)
            if motor is not None:
                self.motors[name] = motor

        self.phase = 0.0
        self.x_amplitude = 0.0
        self.a_amplitude = 0.0
        self.running = False
        self.frequency = 2.4

    def _set_joint(self, name: str, value: float) -> None:
        motor = self.motors.get(name)
        if motor is None:
            return
        sign = self.JOINT_SIGNS.get(name, 1.0)
        target = value * sign
        target = max(motor.getMinPosition(), min(motor.getMaxPosition(), target))
        motor.setPosition(target)

    def set_ready_pose(self) -> None:
        """Postura inicial con plantas apoyadas al 100% y leve inclinación frontal."""
        hip_bend = 0.42
        knee_bend = 0.78
        ankle_bend = 0.40

        self._set_joint("LegUpperR", hip_bend)
        self._set_joint("LegUpperL", hip_bend)
        self._set_joint("LegLowerR", knee_bend)
        self._set_joint("LegLowerL", knee_bend)
        self._set_joint("AnkleR", ankle_bend)
        self._set_joint("AnkleL", ankle_bend)

        self._set_joint("PelvR", 0.0)
        self._set_joint("PelvL", 0.0)
        self._set_joint("FootR", 0.0)
        self._set_joint("FootL", 0.0)
        self._set_joint("PelvYR", 0.0)
        self._set_joint("PelvYL", 0.0)

        self._set_joint("ShoulderR", -0.25)
        self._set_joint("ShoulderL", 0.25)
        self._set_joint("ArmUpperR", -0.35)
        self._set_joint("ArmUpperL", 0.35)
        self._set_joint("ArmLowerR", -1.10)
        self._set_joint("ArmLowerL", 1.10)
        self._set_joint("Neck", 0.0)
        self._set_joint("Head", -0.15)

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False
        self.phase = 0.0
        self.x_amplitude = 0.0
        self.a_amplitude = 0.0
        self.set_ready_pose()

    def setXAmplitude(self, x: float) -> None:
        self.x_amplitude = max(-1.0, min(1.0, x))

    def setAAmplitude(self, a: float) -> None:
        self.a_amplitude = max(-1.0, min(1.0, a))

    def step(self, timestep_ms: int) -> None:
        if not self.running:
            self.set_ready_pose()
            return

        dt = timestep_ms / 1000.0
        self.phase = (self.phase + 2.0 * math.pi * self.frequency * dt) % (2.0 * math.pi)

        s = math.sin(self.phase)
        c = math.cos(self.phase)

        # 1. Balanceo lateral para liberar el pie que da el paso
        sway = -0.12 * s
        self._set_joint("PelvR", sway)
        self._set_joint("PelvL", sway)
        self._set_joint("FootR", -sway)
        self._set_joint("FootL", -sway)

        # 2. Elevación de rodilla y zancada hacia +X
        lift_r = max(0.0, s) * (0.44 + 0.12 * abs(self.x_amplitude))
        lift_l = max(0.0, -s) * (0.44 + 0.12 * abs(self.x_amplitude))

        swing_r = -0.36 * self.x_amplitude * c
        swing_l = 0.36 * self.x_amplitude * c

        hip_r = 0.42 + swing_r + 0.45 * lift_r
        hip_l = 0.42 + swing_l + 0.45 * lift_l
        knee_r = 0.78 + lift_r
        knee_l = 0.78 + lift_l

        # Compensación de tobillo para mantener la planta paralela al suelo
        ankle_r = (knee_r - hip_r) + 0.04
        ankle_l = (knee_l - hip_l) + 0.04

        self._set_joint("LegUpperR", hip_r)
        self._set_joint("LegUpperL", hip_l)
        self._set_joint("LegLowerR", knee_r)
        self._set_joint("LegLowerL", knee_l)
        self._set_joint("AnkleR", ankle_r)
        self._set_joint("AnkleL", ankle_l)

        # 3. Giro en caderas
        yaw = 0.18 * self.a_amplitude * s
        self._set_joint("PelvYR", yaw)
        self._set_joint("PelvYL", yaw)

        # 4. Braceo alterno
        self._set_joint("ShoulderR", -0.25 + 0.35 * self.x_amplitude * c)
        self._set_joint("ShoulderL", 0.25 - 0.35 * self.x_amplitude * c)