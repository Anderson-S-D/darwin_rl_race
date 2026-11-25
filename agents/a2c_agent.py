from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
from agents.base_agent import BaseAgent


class SharedActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.actor_head(features), self.critic_head(features)


class A2CAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 7e-4,
        gamma: float = 0.99,
        n_steps: int = 32,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        super().__init__(state_dim, action_dim)
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.model = SharedActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(state_t)
            if eval_mode:
                return int(torch.argmax(logits, dim=1).item())
            dist = Categorical(logits=logits)
            return int(dist.sample().item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def update(self) -> Optional[float]:
        if not self.buffer:
            return None

        last_done = self.buffer[-1][4]
        if len(self.buffer) < self.n_steps and not last_done:
            return None

        states, actions, rewards, next_states, dones = zip(*self.buffer)
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)

        with torch.no_grad():
            if last_done:
                next_value = 0.0
            else:
                last_next_t = torch.as_tensor(
                    next_states[-1], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                _, val = self.model(last_next_t)
                next_value = float(val.item())

        returns = []
        R = next_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1.0 - float(d))
            returns.insert(0, R)

        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        logits, values = self.model(states_t)
        values = values.squeeze(-1)

        advantages = returns_t - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.functional.mse_loss(values, returns_t)
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        self.buffer.clear()
        return float(loss.item())