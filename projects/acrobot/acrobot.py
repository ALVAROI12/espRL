"""Acrobot helpers for the on-chip run: the env itself is Gymnasium's Acrobot-v1 (physics port in acrobot.c)."""

from __future__ import annotations

import numpy as np


def obs_of(state) -> np.ndarray:
    """Observation from the state [th1, th2, dth1, dth2], as Acrobot-v1 builds it."""
    t1, t2, d1, d2 = state
    return np.array([np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2), d1, d2], dtype=np.float32)


def step_reward(state, action, terminated: bool) -> float:
    """Acrobot-v1: -1 per step, 0 on the step that reaches the goal."""
    return 0.0 if terminated else -1.0
