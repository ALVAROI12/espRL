"""Continuous cart-pole (balance or swing-up) for SAC; design in docs/superpowers/specs/2026-11-16-cartpole-design.md.

Frictionless classic cart-pole (Florian 2007), semi-implicit Euler at 50 Hz. th = 0 is upright.
Obs [x, x_dot, cos th, sin th, th_dot]; action in [-1, 1] is force / FORCE_MAX. Registered as espRL/CartPole-v0
(500 steps); pick the task with gym.make("espRL/CartPole-v0", mode="balance" | "swingup").
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

M_CART, M_POLE, HALF_LEN, G = 1.0, 0.1, 0.5, 9.81
DT, FORCE_MAX, X_MAX = 0.02, 10.0, 2.4
OUT_OF_TRACK = -100.0
# Reward weights (tunable; the success test in train.py is not).
W_X, W_U = 0.1, 0.001


class CartPole(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mode: str = "balance"):
        assert mode in ("balance", "swingup"), mode
        self.mode = mode
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        hi = np.array([np.inf] * 5, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-hi, hi, dtype=np.float32)
        self.state = np.zeros(4)

    def _obs(self) -> np.ndarray:
        x, xd, th, thd = self.state
        return np.array([x, xd, np.cos(th), np.sin(th), thd], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(-0.05, 0.05, 4)
        if self.mode == "swingup":
            self.state[2] += np.pi
        return self._obs(), {}

    def step(self, action):
        u = FORCE_MAX * float(np.clip(action[0], -1.0, 1.0))
        x, xd, th, thd = self.state
        s, c = np.sin(th), np.cos(th)
        m = M_CART + M_POLE
        temp = (u + M_POLE * HALF_LEN * thd * thd * s) / m
        th_acc = (G * s - c * temp) / (HALF_LEN * (4.0 / 3.0 - M_POLE * c * c / m))
        x_acc = temp - M_POLE * HALF_LEN * th_acc * c / m
        xd += DT * x_acc
        thd += DT * th_acc
        x += DT * xd
        th += DT * thd
        self.state = np.array([x, xd, th, thd])
        if abs(x) > X_MAX:
            return self._obs(), OUT_OF_TRACK, True, False, {}
        return self._obs(), float(np.cos(th) - W_X * x * x - W_U * u * u), False, False, {}


gym.register("espRL/CartPole-v0", entry_point=CartPole, max_episode_steps=500)
