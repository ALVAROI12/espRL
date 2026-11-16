"""mlp.c (the code that runs on the ESP32) must match the SB3 torch actor it was exported from."""

import ctypes
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import SAC

import cartpole  # noqa: F401  (registers espRL/CartPole-v0)
from export import export_header, header_env, reference_action

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


def pendulum_obs(rng, n):
    th, thd = rng.uniform(-np.pi, np.pi, n), rng.uniform(-8, 8, n)
    return np.stack([np.cos(th), np.sin(th), thd], 1)


def cartpole_obs(rng, n):
    th = rng.uniform(-np.pi, np.pi, n)
    return np.stack([rng.uniform(-2.4, 2.4, n), rng.uniform(-5, 5, n), np.cos(th), np.sin(th),
                     rng.uniform(-10, 10, n)], 1)


@pytest.mark.parametrize("name,env_id,obs_fn", [("pendulum", "Pendulum-v1", pendulum_obs),
                                                ("cartpole_balance", "espRL/CartPole-v0", cartpole_obs)])
def test_c_forward_matches_torch(tmp_path, name, env_id, obs_fn):
    model = SAC("MlpPolicy", gym.make(env_id), policy_kwargs=dict(net_arch=[64, 64]), seed=0)  # untrained
    header = tmp_path / "policy_weights.h"
    export_header(model, header, name)
    assert header_env(header) == name
    lib = build_mlp(tmp_path)
    obs = obs_fn(np.random.default_rng(0), 1000).astype(np.float32)
    assert ctypes.c_int.in_dll(lib, "mlp_n_obs").value == obs.shape[1]
    err = max(abs(c_action(lib, o) - reference_action(model, o)) for o in obs)
    assert err < 1e-5
