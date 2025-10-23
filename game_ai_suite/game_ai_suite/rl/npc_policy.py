from __future__ import annotations
from typing import Optional
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from .gridworld import GridWorldEnv


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "npc_ppo.zip")


def train_npc(total_timesteps: int = 5000) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    env = DummyVecEnv([lambda: GridWorldEnv(size=5)])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(MODEL_PATH)
    return MODEL_PATH


def load_npc() -> PPO:
    return PPO.load(MODEL_PATH)


def run_npc_episode(model: PPO, render: bool = False) -> float:
    env = GridWorldEnv(size=5)
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(int(action))
        total_reward += float(reward)
        if render:
            env.render()
    return total_reward
