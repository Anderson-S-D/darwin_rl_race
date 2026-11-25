from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import torch
import torch.nn as nn


class BaseAgent(ABC):
    def __init__(self, state_dim: int, action_dim: int, device: Optional[str] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: nn.Module

    @abstractmethod
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        pass

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        pass

    @abstractmethod
    def update(self) -> Optional[float]:
        pass

    def save_weights(self, filepath: str) -> None:
        torch.save(self.model.state_dict(), filepath)

    def load_weights(self, filepath: str) -> None:
        state_dict = torch.load(filepath, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()