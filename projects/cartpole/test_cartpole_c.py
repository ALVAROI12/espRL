"""cartpole.c (physics on the ESP32, float32) must match cartpole.py (float64) step by step."""

import ctypes
import subprocess
from pathlib import Path

import numpy as np

from projects.cartpole.cartpole import CartPole

HERE = Path(__file__).resolve().parent
F4 = ctypes.c_float * 4


def build(tmp_path) -> ctypes.CDLL:
    lib = tmp_path / "libcartpole.so"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Werror", "-shared", "-fPIC", str(HERE / "cartpole.c"),
                    "-o", str(lib), "-lm"], check=True)
    dll = ctypes.CDLL(str(lib))
    dll.cartpole_step.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_float]
    dll.cartpole_step.restype = ctypes.c_int
    dll.cartpole_obs.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    return dll


def test_c_step_matches_python(tmp_path):
    lib = build(tmp_path)
    rng = np.random.default_rng(0)
    env = CartPole()
    env.reset(seed=0)
    for _ in range(2000):
        s = np.array([rng.uniform(-2.4, 2.4), rng.uniform(-5, 5), rng.uniform(-7, 7), rng.uniform(-10, 10)],
                     dtype=np.float32)
        a = np.float32(rng.uniform(-1.5, 1.5))
        env.state = s.astype(float)
        obs, _, terminated, _, _ = env.step(np.array([a]))
        c = F4(*s)
        c_obs = (ctypes.c_float * 5)()
        lib.cartpole_obs(c, c_obs)
        x, xd, th, thd = s.astype(float)
        assert np.allclose(np.array(c_obs), [x, xd, np.cos(th), np.sin(th), thd], atol=1e-6)
        c_term = lib.cartpole_step(c, a)
        assert np.allclose(np.array(c), env.state, rtol=1e-5, atol=1e-5)
        assert bool(c_term) == terminated
