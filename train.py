"""Train SAC on Pendulum-v1, save out/sac_pendulum.zip and export firmware/main/policy_weights.h.

Usage: python train.py [timesteps]   (exit 1 if the deterministic mean return over 10 episodes is <= -250)
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

from export import export_header

RL = Path(__file__).resolve().parent
MODEL = RL / "out" / "sac_pendulum.zip"
HEADER = RL / "firmware" / "main" / "policy_weights.h"
MIN_RETURN = -250.0


def evaluate(model: SAC, episodes: int = 10, seed: int = 0) -> list[float]:
    env = gym.make("Pendulum-v1")
    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    return returns


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    model = SAC("MlpPolicy", gym.make("Pendulum-v1"), policy_kwargs=dict(net_arch=[64, 64]), seed=0, verbose=0)
    model.learn(total_timesteps=steps)
    MODEL.parent.mkdir(exist_ok=True)
    model.save(MODEL)
    export_header(model, HEADER, "pendulum")
    mean = float(np.mean(evaluate(model)))
    print(f"timesteps {steps}  mean return {mean:.1f}  (gate > {MIN_RETURN})")
    return 0 if mean > MIN_RETURN else 1


if __name__ == "__main__":
    sys.exit(main())
