from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
from agents.base_agent import BaseAgent


class ActorCriticSeparated(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(x), self.critic(x)


class PPOAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        rollout_steps: int = 256,
        epochs: int = 4,
        mini_batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        super().__init__(state_dim, action_dim)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.rollout_steps = rollout_steps
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.model = ActorCriticSeparated(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self._last_log_prob: float = 0.0
        self._last_value: float = 0.0
        self.buffer: List[Tuple[np.ndarray, int, float, float, float, np.ndarray, bool]] = []

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state_t)
            if eval_mode:
                return int(torch.argmax(logits, dim=1).item())
            dist = Categorical(logits=logits)
            action = dist.sample()
            self._last_log_prob = float(dist.log_prob(action).item())
            self._last_value = float(value.item())
            return int(action.item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            (state, action, self._last_log_prob, self._last_value, reward, next_state, done)
        )

    def update(self) -> Optional[float]:
        if len(self.buffer) < self.rollout_steps:
            return None

        states, actions, old_log_probs, values, rewards, next_states, dones = zip(*self.buffer)

        with torch.no_grad():
            last_next_t = torch.as_tensor(
                next_states[-1], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            _, last_val_t = self.model(last_next_t)
            next_value = 0.0 if dones[-1] else float(last_val_t.item())

        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - float(dones[t])
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae

        returns = advantages + np.array(values, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        total_loss = 0.0
        update_count = 0
        indices = np.arange(len(self.buffer))

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, len(self.buffer), self.mini_batch_size):
                mb_idx = indices[start : start + self.mini_batch_size]
                logits, curr_values = self.model(states_t[mb_idx])
                curr_values = curr_values.squeeze(-1)

                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t[mb_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[mb_idx])
                surr1 = ratio * advantages_t[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t[mb_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(curr_values, returns_t[mb_idx])
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_loss += float(loss.item())
                update_count += 1

        self.buffer.clear()
        return total_loss / max(update_count, 1)