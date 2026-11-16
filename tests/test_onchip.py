"""onchip.py checks against a fake chip that runs cartpole.py and the torch actor (no board needed)."""

import numpy as np
from stable_baselines3 import SAC

import onchip
import train
from projects.cartpole.cartpole import CartPole
from export import reference_action


class FakeChip:
    """Answers start_episode/frame like the firmware; drop = step index whose frame never arrives."""

    def __init__(self, model, drop=None, period_us=20_000):
        self.model, self.drop, self.period_us, self.frames = model, drop, period_us, []

    def start_episode(self, state):
        env = CartPole()
        env.reset(seed=0)
        env.state = np.asarray(state, dtype=float)
        self.frames = []
        for k in range(500):
            a = float(reference_action(self.model, env._obs())[0])
            _, _, term, _, _ = env.step(np.array([a]))
            flags = int(term) | (2 if k == 499 else 0)
            if k != self.drop:
                self.frames.append((k, tuple(np.float32(env.state)), a, 25_000, k * self.period_us, flags))
            if term:
                break
        return 0

    def frame(self, seq, timeout=None):
        return self.frames.pop(0) if self.frames else None


def model():
    return SAC("MlpPolicy", train.make_env("cartpole_balance"), policy_kwargs=dict(net_arch=[64, 64]), seed=0)


def test_clean_run_meets_link_gates():
    rep = onchip.run(FakeChip(m := model()), m, "cartpole_balance", episodes=2)
    assert rep["incomplete"] == 0 and rep["max_abs_action_diff"] < 1e-4
    assert rep["max_period_error_us"] == 0 and rep["steps"] >= 2


def test_dropped_frame_fails():
    m = model()
    rep = onchip.run(FakeChip(m, drop=0), m, "cartpole_balance", episodes=1)
    assert rep["incomplete"] == 1 and not rep["passed"]


def test_late_steps_fail():
    m = model()
    rep = onchip.run(FakeChip(m, period_us=21_500), m, "cartpole_balance", episodes=1)
    assert rep["max_period_error_us"] == 1_500 and not rep["passed"]
