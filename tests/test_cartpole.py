"""Physics and episode rules of the cart-pole env (cartpole.py)."""

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import cartpole  # noqa: F401  (registers espRL/CartPole-v0)
from cartpole import CartPole

ZERO = np.zeros(1, dtype=np.float32)


def at(state, mode="balance"):
    env = CartPole(mode)
    env.reset(seed=0)
    env.state = np.array(state, dtype=float)
    return env


def test_hanging_at_rest_stays():
    env = at([0, 0, np.pi, 0])
    for _ in range(100):
        env.step(ZERO)
    x, _, th, _ = env.state
    assert abs(th - np.pi) < 1e-6 and abs(x) < 1e-6


def test_upright_tilt_falls():
    env = at([0, 0, 0.01, 0])
    for _ in range(100):
        env.step(ZERO)
    assert abs(env.state[2]) > 0.5


def test_force_pushes_cart():
    env = at([0, 0, np.pi, 0])
    env.step(np.ones(1, dtype=np.float32))
    assert env.state[1] > 0
    env = at([0, 0, np.pi, 0])
    env.step(-np.ones(1, dtype=np.float32))
    assert env.state[1] < 0


def test_track_exit_terminates():
    env = at([2.39, 5, 0, 0])
    _, reward, terminated, _, _ = env.step(ZERO)
    assert terminated and reward == -100


def test_obs_periodic_in_theta():
    a = at([0.3, 0.1, np.pi, 0.2])._obs()
    b = at([0.3, 0.1, 5 * np.pi, 0.2])._obs()
    assert np.allclose(a, b, atol=1e-6)


def test_reset_modes():
    for mode, center in (("balance", 0.0), ("swingup", np.pi)):
        env = CartPole(mode)
        for s in range(20):
            obs, _ = env.reset(seed=s)
            assert obs.shape == (5,) and obs.dtype == np.float32
            assert abs(env.state[2] - center) <= 0.05
        check_env(CartPole(mode), skip_render_check=True)


def test_truncates_at_500():
    env = gym.make("espRL/CartPole-v0")
    env.reset(seed=0)
    env.unwrapped.state = np.array([0, 0, np.pi, 0], dtype=float)
    for i in range(1, 501):
        _, _, terminated, truncated, _ = env.step(ZERO)
        assert not terminated
        assert truncated == (i == 500)
