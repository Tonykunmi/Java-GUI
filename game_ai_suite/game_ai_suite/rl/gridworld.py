from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, size: int = 5) -> None:
        super().__init__()
        self.size = size
        self.observation_space = spaces.Box(0, size - 1, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)  # 0:up,1:right,2:down,3:left
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([size - 1, size - 1], dtype=np.int32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        return self.agent_pos.copy(), {}

    def step(self, action: int):
        if action == 0:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 3:
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)

        done = bool(np.all(self.agent_pos == self.goal_pos))
        reward = 1.0 if done else -0.01
        return self.agent_pos.copy(), reward, done, False, {}

    def render(self):
        grid = np.full((self.size, self.size), fill_value=".", dtype=object)
        grid[tuple(self.goal_pos)] = "G"
        grid[tuple(self.agent_pos)] = "A"
        print("\n".join(" ".join(row) for row in grid))
