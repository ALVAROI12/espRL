"""rl/mlp.c (the code that runs on the ESP32) must match the SB3 torch actor it was exported from."""

import ctypes
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

from export import export_header, reference_action

RL = Path(__file__).resolve().parents[1]


def build_mlp(include_dir: Path) -> ctypes.CDLL:
    lib = include_dir / "libmlp.so"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Werror", "-shared", "-fPIC", f"-I{include_dir}",
                    f"-I{RL}", str(RL / "mlp.c"), "-o", str(lib), "-lm"], check=True)
    dll = ctypes.CDLL(str(lib))
    dll.mlp_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    dll.mlp_forward.restype = ctypes.c_float
    return dll


def c_action(lib: ctypes.CDLL, obs: np.ndarray) -> float:
    o = np.ascontiguousarray(obs, dtype=np.float32)
    return lib.mlp_forward(o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))


def test_c_forward_matches_torch(tmp_path):
    model = SAC("MlpPolicy", gym.make("Pendulum-v1"), policy_kwargs=dict(net_arch=[64, 64]), seed=0)  # untrained
    export_header(model, tmp_path / "policy_weights.h")
    lib = build_mlp(tmp_path)
    rng = np.random.default_rng(0)
    th = rng.uniform(-np.pi, np.pi, 1000)
    thd = rng.uniform(-8, 8, 1000)
    obs = np.stack([np.cos(th), np.sin(th), thd], 1).astype(np.float32)
    err = max(abs(c_action(lib, o) - reference_action(model, o)) for o in obs)
    assert err < 1e-5
