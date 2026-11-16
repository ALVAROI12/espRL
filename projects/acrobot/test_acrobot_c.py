"""acrobot.c (physics on the ESP32, float32) must match Gymnasium's Acrobot-v1 (float64) step by step."""

import ctypes
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np

from projects.acrobot.acrobot import obs_of, step_reward

HERE = Path(__file__).resolve().parent
F4 = ctypes.c_float * 4


def build(tmp_path) -> ctypes.CDLL:
    lib = tmp_path / "libacrobot.so"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Werror", "-shared", "-fPIC", str(HERE / "acrobot.c"),
                    "-o", str(lib), "-lm"], check=True)
    dll = ctypes.CDLL(str(lib))
    dll.acrobot_step.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_float]
    dll.acrobot_step.restype = ctypes.c_int
    dll.acrobot_obs.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    return dll


def angle_diff(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi)


def test_c_step_matches_gymnasium(tmp_path):
    lib = build(tmp_path)
    env = gym.make("Acrobot-v1").unwrapped
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(3000):
        s = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi), rng.uniform(-4 * np.pi, 4 * np.pi),
                      rng.uniform(-9 * np.pi, 9 * np.pi)], dtype=np.float32)
        a = int(rng.integers(0, 3))
        env.state = s.astype(float)
        obs, reward, terminated, _, _ = env.step(a)
        c = F4(*s)
        c_obs = (ctypes.c_float * 6)()
        lib.acrobot_obs(c, c_obs)
        assert np.allclose(np.array(c_obs), obs_of(s), atol=1e-6)
        c_term = lib.acrobot_step(c, float(a))
        cs = np.array(c)
        err = max(angle_diff(cs[:2], env.state[:2]).max(), np.abs(cs[2:] - env.state[2:]).max())
        errs.append(err)
        # float32 near the velocity bounds (|dth2| ~ 9 pi): up to ~3e-3 measured; typical states ~1e-6.
        assert err < 5e-3
        if abs(-np.cos(env.state[0]) - np.cos(env.state[0] + env.state[1]) - 1.0) > 1e-4:  # not on the goal line
            assert bool(c_term) == terminated
        assert step_reward(cs, a, terminated) == reward
        assert np.allclose(obs_of(env.state), obs)
    assert np.median(errs) < 1e-5 and np.percentile(errs, 99) < 1e-4
