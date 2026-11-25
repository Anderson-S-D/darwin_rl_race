import gymnasium as gym
from gymnasium import spaces
import numpy as np
from controller import Supervisor
from gait_controller import DarwinGaitController


class DarwinRaceEnv(gym.Env):
    def __init__(self, robot: Supervisor, goal_x: float = 10.0):
        super().__init__()
        self.robot = robot
        self.timestep = int(self.robot.getBasicTimeStep())
        self.goal_x = goal_x

        self.robot_node = self.robot.getSelf()
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")
        self.initial_trans = list(self.trans_field.getSFVec3f())
        self.initial_rot = list(self.rot_field.getSFRotation())

        # Sensores integrados y opcionales
        self.acc = self.robot.getDevice("Accelerometer")
        if self.acc:
            self.acc.enable(self.timestep)

        self.gyro = self.robot.getDevice("Gyro")
        if self.gyro:
            self.gyro.enable(self.timestep)

        self.ds_left = self.robot.getDevice("ds_left")
        if self.ds_left:
            self.ds_left.enable(self.timestep)

        self.ds_right = self.robot.getDevice("ds_right")
        if self.ds_right:
            self.ds_right.enable(self.timestep)

        self.gait_manager = DarwinGaitController(self.robot)
        self.gait_manager.set_ready_pose()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.prev_x = self.initial_trans[0]

    def _get_position(self) -> list:
        return list(self.trans_field.getSFVec3f())

    def _get_state(self) -> np.ndarray:
        pos = self._get_position()
        acc = self.acc.getValues() if self.acc else [0.0, 0.0, 9.81]
        gyro = self.gyro.getValues() if self.gyro else [0.0, 0.0, 0.0]

        d_left = min(self.ds_left.getValue() / 1000.0, 1.0) if self.ds_left else 1.0
        d_right = min(self.ds_right.getValue() / 1000.0, 1.0) if self.ds_right else 1.0

        dist_goal = np.clip((self.goal_x - pos[0]) / self.goal_x, -1.0, 1.0)
        lateral_offset = np.clip(pos[1] / 2.0, -1.0, 1.0)

        return np.array([
            dist_goal,
            lateral_offset,
            np.clip(acc[0] / 20.0, -1.0, 1.0),
            np.clip(acc[2] / 20.0, -1.0, 1.0),
            np.clip(gyro[1] / 10.0, -1.0, 1.0),
            d_left,
            d_right,
            1.0 if pos[2] > 0.22 else -1.0,
        ], dtype=np.float32)

    def _apply_action(self, action: int) -> None:
        self.gait_manager.start()
        if action == 0:    # Avance rápido
            self.gait_manager.setXAmplitude(1.0)
            self.gait_manager.setAAmplitude(0.0)
        elif action == 1:  # Avance prudente
            self.gait_manager.setXAmplitude(0.55)
            self.gait_manager.setAAmplitude(0.0)
        elif action == 2:  # Girar izquierda
            self.gait_manager.setXAmplitude(0.45)
            self.gait_manager.setAAmplitude(0.6)
        elif action == 3:  # Girar derecha
            self.gait_manager.setXAmplitude(0.45)
            self.gait_manager.setAAmplitude(-0.6)
        elif action == 4:  # Paso estabilizador
            self.gait_manager.setXAmplitude(0.15)
            self.gait_manager.setAAmplitude(0.0)

        for _ in range(12):
            self.gait_manager.step(self.timestep)
            if self.robot.step(self.timestep) == -1:
                break

    def step(self, action: int):
        self._apply_action(action)
        state = self._get_state()
        pos = self._get_position()

        progress = (pos[0] - self.prev_x) * 15.0
        self.prev_x = pos[0]

        min_sensor = min(state[5], state[6])
        obstacle_penalty = -1.5 if min_sensor < 0.25 else 0.0

        fallen = pos[2] < 0.20
        reached_goal = pos[0] >= self.goal_x

        reward = progress + obstacle_penalty - 0.04
        if fallen:
            reward -= 20.0
        if reached_goal:
            reward += 50.0

        done = bool(fallen or reached_goal)
        return state, float(reward), done, False, {"x": pos[0], "fallen": fallen}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gait_manager.stop()
        self.trans_field.setSFVec3f(self.initial_trans)
        self.rot_field.setSFRotation(self.initial_rot)
        self.robot_node.resetPhysics()

        for _ in range(15):
            self.gait_manager.set_ready_pose()
            if self.robot.step(self.timestep) == -1:
                break

        self.prev_x = self.initial_trans[0]
        return self._get_state(), {}