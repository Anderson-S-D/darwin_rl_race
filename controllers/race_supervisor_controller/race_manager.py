import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAINER_CTRL_DIR = os.path.join(PROJECT_ROOT, "controllers", "rl_trainer_controller")

for path in (PROJECT_ROOT, TRAINER_CTRL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from controller import Supervisor
from gait_controller import DarwinGaitController
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent

MODEL_CONFIG = {
    "darwin_ppo": ("ppo", PPOAgent, ["weights/ppo_ep350.pth", "weights/ppo_ep250.pth"]),
    "darwin_dqn": ("dqn", DQNAgent, ["weights/dqn_ep350.pth", "weights/dqn_ep250.pth"]),
    "darwin_a2c": ("a2c", A2CAgent, ["weights/a2c_ep350.pth", "weights/a2c_ep250.pth"]),
}


def resolve_checkpoint(candidates):
    for rel_path in candidates:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if os.path.exists(full_path):
            return full_path
    return None


def get_observation(robot: Supervisor, trans_field, acc, gyro, ds_left, ds_right, goal_x: float) -> np.ndarray:
    pos = list(trans_field.getSFVec3f())
    a_vals = acc.getValues() if acc else [0.0, 0.0, 9.81]
    g_vals = gyro.getValues() if gyro else [0.0, 0.0, 0.0]
    d_left = min(ds_left.getValue() / 1000.0, 1.0) if ds_left else 1.0
    d_right = min(ds_right.getValue() / 1000.0, 1.0) if ds_right else 1.0

    dist_goal = np.clip((goal_x - pos[0]) / goal_x, -1.0, 1.0)
    lateral_offset = np.clip(pos[1] / 2.0, -1.0, 1.0)

    return np.array([
        dist_goal,
        lateral_offset,
        np.clip(a_vals[0] / 20.0, -1.0, 1.0),
        np.clip(a_vals[2] / 20.0, -1.0, 1.0),
        np.clip(g_vals[1] / 10.0, -1.0, 1.0),
        d_left,
        d_right,
        1.0 if pos[2] > 0.22 else -1.0,
    ], dtype=np.float32)


def apply_action(gait: DarwinGaitController, action: int) -> None:
    gait.start()
    if action == 0:
        gait.setXAmplitude(1.0)
        gait.setAAmplitude(0.0)
    elif action == 1:
        gait.setXAmplitude(0.55)
        gait.setAAmplitude(0.0)
    elif action == 2:
        gait.setXAmplitude(0.45)
        gait.setAAmplitude(0.6)
    elif action == 3:
        gait.setXAmplitude(0.45)
        gait.setAAmplitude(-0.6)
    elif action == 4:
        gait.setXAmplitude(0.15)
        gait.setAAmplitude(0.0)


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    goal_x = 10.0

    robot_name = robot.getName()
    algo_key, agent_cls, ckpt_candidates = MODEL_CONFIG.get(
        robot_name,
        ("ppo", PPOAgent, ["weights/ppo_ep350.pth", "weights/ppo_ep250.pth"]),
    )

    agent = agent_cls(state_dim=8, action_dim=5)
    ckpt_path = resolve_checkpoint(ckpt_candidates)
    if ckpt_path:
        agent.load_weights(ckpt_path)
        print(f"[{robot_name} | {algo_key.upper()}] Pesos cargados desde: {ckpt_path}")
    else:
        print(f"[{robot_name} | {algo_key.upper()}] Sin checkpoint encontrado, usando pesos iniciales.")

    robot_node = robot.getSelf()
    trans_field = robot_node.getField("translation")

    acc = robot.getDevice("Accelerometer")
    if acc:
        acc.enable(timestep)
    gyro = robot.getDevice("Gyro")
    if gyro:
        gyro.enable(timestep)
    ds_left = robot.getDevice("ds_left")
    if ds_left:
        ds_left.enable(timestep)
    ds_right = robot.getDevice("ds_right")
    if ds_right:
        ds_right.enable(timestep)

    gait = DarwinGaitController(robot)
    gait.set_ready_pose()

    finished = False
    start_time = robot.getTime()
    step_counter = 0
    current_action = 1

    while robot.step(timestep) != -1:
        pos = list(trans_field.getSFVec3f())
        fallen = pos[2] < 0.20

        if not finished and pos[0] >= goal_x:
            finished = True
            elapsed = robot.getTime() - start_time
            print(f"[META] {robot_name} ({algo_key.upper()}) cruzó la meta en {elapsed:.2f}s (X={pos[0]:.2f}m)")
            gait.stop()
            continue

        if fallen:
            gait.stop()
            continue

        if step_counter % 12 == 0:
            obs = get_observation(robot, trans_field, acc, gyro, ds_left, ds_right, goal_x)
            current_action = agent.select_action(obs, eval_mode=True)
            apply_action(gait, current_action)

        gait.step(timestep)
        step_counter += 1


if __name__ == "__main__":
    main()