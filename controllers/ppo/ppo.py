import csv
import os
import re
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from controller import Supervisor
from darwin_env import DarwinRaceEnv
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent

AGENT_FACTORY = {
    "dqn": DQNAgent,
    "a2c": A2CAgent,
    "ppo": PPOAgent,
}

# Cambia aquí el algoritmo a reentrenar: "ppo", "dqn" o "a2c"
SELECTED_ALGORITHM = "ppo"
EXTRA_EPISODES = 100
SAVE_EVERY = 50

# Ruta relativa a la raíz del proyecto (ej. "weights/ppo_ep250.pth") o None para empezar de cero
RESUME_CHECKPOINT = f"weights/{SELECTED_ALGORITHM}_ep650.pth"


def _extract_episode_from_path(ckpt_path: str) -> int:
    match = re.search(r"_ep(\d+)\.pth$", os.path.basename(ckpt_path))
    return int(match.group(1)) if match else 0


def main():
    robot = Supervisor()
    env = DarwinRaceEnv(robot, goal_x=10.0)

    agent_cls = AGENT_FACTORY[SELECTED_ALGORITHM]
    agent = agent_cls(state_dim=8, action_dim=5)

    weights_dir = os.path.join(PROJECT_ROOT, "weights")
    metrics_dir = os.path.join(PROJECT_ROOT, "metrics")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    start_episode = 1
    if RESUME_CHECKPOINT:
        ckpt_path = os.path.join(PROJECT_ROOT, RESUME_CHECKPOINT)
        if os.path.exists(ckpt_path):
            agent.load_weights(ckpt_path)
            # Reabrir en modo entrenamiento tras load_weights (que llama a .eval())
            agent.model.train()
            prev_ep = _extract_episode_from_path(ckpt_path)
            start_episode = prev_ep + 1
            # En reentrenamiento con obstáculos reducimos exploración aleatoria para no perder el equilibrio
            if hasattr(agent, "epsilon"):
                agent.epsilon = max(getattr(agent, "epsilon_end", 0.05), 0.15)
            print(f"[{SELECTED_ALGORITHM.upper()}] Reanudando desde {ckpt_path} (Episodio {start_episode})")
        else:
            print(f"[{SELECTED_ALGORITHM.upper()}] Checkpoint no encontrado ({ckpt_path}), iniciando desde 0.")

    end_episode = start_episode + EXTRA_EPISODES - 1
    csv_path = os.path.join(metrics_dir, f"{SELECTED_ALGORITHM}_training.csv")
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    write_mode = "a" if (RESUME_CHECKPOINT and file_exists) else "w"

    with open(csv_path, mode=write_mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if write_mode == "w":
            writer.writerow(["episode", "reward", "distance_m", "steps", "fallen", "avg_loss"])

        for episode in range(start_episode, end_episode + 1):
            state, _ = env.reset()
            total_reward = 0.0
            steps = 0
            losses = []
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)

                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                total_reward += reward
                steps += 1

            avg_loss = sum(losses) / len(losses) if losses else 0.0
            writer.writerow([
                episode,
                round(total_reward, 3),
                round(info["x"], 3),
                steps,
                int(info["fallen"]),
                round(avg_loss, 5),
            ])
            csv_file.flush()

            print(
                f"[{SELECTED_ALGORITHM.upper()}] Ep {episode:03d}/{end_episode} | "
                f"Recompensa: {total_reward:7.2f} | Distancia: {info['x']:5.2f}m | "
                f"Caída: {info['fallen']} | Loss: {avg_loss:.4f}"
            )

            if episode % SAVE_EVERY == 0 or episode == end_episode:
                save_path = os.path.join(weights_dir, f"{SELECTED_ALGORITHM}_ep{episode}.pth")
                agent.save_weights(save_path)
                if RESUME_CHECKPOINT:
                    agent.model.train()


if __name__ == "__main__":
    main()