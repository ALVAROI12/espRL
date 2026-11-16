"""Train a policy (SAC, or DQN for discrete actions) on one env, save out/<env>.zip and export
firmware/main/policy_weights.h.

Usage: python train.py ENV [timesteps]   ENV: pendulum | cartpole_balance | cartpole_swingup | lunarlander | acrobot
Exit 1 if the deterministic policy misses the env's gate over 10 episodes (gate() below).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

import projects.acrobot.acrobot as acrobot_sim
import projects.cartpole.cartpole as cartpole_sim  # also registers espRL/CartPole-v0
from export import export_header, reference_action

RL = Path(__file__).resolve().parent
OUT = RL / "out"
HEADER = RL / "firmware" / "main" / "policy_weights.h"
SUCCESS_RATE = 0.9  # cart-pole gate: 9 of 10 episodes
UPRIGHT_COS, UPRIGHT_STEPS = 0.95, 100  # cart-pole success: cos th > 0.95 on each of the last 2 s

# steps: default timesteps; algo: "sac" (default) or "dqn"; hp: algorithm settings besides the defaults;
# min_return: gate on the mean return (else the cart-pole success gate); cos: index of cos th in the observation
# (success test); fps: 1 / env dt (viz playback, on-chip step period); render: viz draws env.render();
# sim: Python helpers (obs_of, step_reward) of an env whose physics also runs on the chip (make onchip).
ENVS = {
    "pendulum": dict(id="Pendulum-v1", kwargs={}, steps=20_000, min_return=-250.0, cos=0, fps=20),
    "cartpole_balance": dict(id="espRL/CartPole-v0", kwargs={"mode": "balance"}, steps=60_000, cos=2, fps=50,
                             sim=cartpole_sim),
    "cartpole_swingup": dict(id="espRL/CartPole-v0", kwargs={"mode": "swingup"}, steps=200_000, cos=2, fps=50,
                             sim=cartpole_sim),
    # Gate 200 = Gymnasium's "solved". SAC settings from rl-baselines3-zoo (LunarLanderContinuous).
    "lunarlander": dict(id="LunarLander-v3", kwargs={"continuous": True}, steps=300_000, min_return=200.0, fps=50,
                        render=True, hp=dict(learning_rate=7.3e-4, buffer_size=1_000_000, batch_size=256, tau=0.01,
                                         learning_starts=10_000)),
    # Gate -100 = Acrobot-v1's reward_threshold. DQN settings from rl-baselines3-zoo (Acrobot-v1), net [64, 64].
    "acrobot": dict(id="Acrobot-v1", kwargs={}, steps=100_000, algo="dqn", min_return=-100.0, fps=5, render=True,
                    sim=acrobot_sim,
                    hp=dict(learning_rate=6.3e-4, batch_size=128, buffer_size=50_000, learning_starts=0,
                            target_update_interval=250, train_freq=4, gradient_steps=-1, exploration_fraction=0.12,
                            exploration_final_eps=0.1)),
}
ALGOS = {"sac": SAC, "dqn": DQN}


def check_env_name(name: str) -> None:
    if name not in ENVS:
        print(f"unknown env {name!r}; use one of: {', '.join(ENVS)}", file=sys.stderr)
        sys.exit(2)


def make_env(name: str, **extra) -> gym.Env:
    return gym.make(ENVS[name]["id"], **ENVS[name]["kwargs"], **extra)


def model_path(name: str) -> Path:
    return OUT / f"{name}.zip"


def load(name: str) -> BaseAlgorithm:
    return ALGOS[ENVS[name].get("algo", "sac")].load(model_path(name))


def success(cos_hist: list[float], terminated: bool) -> bool:
    """Cart-pole success: never left the track and upright for the whole last UPRIGHT_STEPS steps."""
    return not terminated and len(cos_hist) >= UPRIGHT_STEPS and min(cos_hist[-UPRIGHT_STEPS:]) > UPRIGHT_COS


def rollout(name: str, env: gym.Env, act: Callable[[np.ndarray], np.ndarray], seed: int) -> tuple[float, bool]:
    """One episode with act(obs) -> action(s); returns (return, success). Success is False without a cos index."""
    obs, _ = env.reset(seed=seed)
    total, cos_hist, terminated, truncated = 0.0, [], False, False
    cos = ENVS[name].get("cos")
    disc = isinstance(env.action_space, gym.spaces.Discrete)
    while not (terminated or truncated):
        a = np.asarray(act(obs), dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, _ = env.step(int(a[0]) if disc else a)
        total += float(reward)
        if cos is not None:
            cos_hist.append(float(obs[cos]))
    return total, cos is not None and success(cos_hist, terminated)


def gate(name: str, returns: list[float], successes: list[bool]) -> tuple[bool, str]:
    mean = float(np.mean(returns))
    if "min_return" in ENVS[name]:
        lo = ENVS[name]["min_return"]
        return mean > lo, f"mean return {mean:.1f}  gate > {lo}"
    n = sum(successes)
    return n >= SUCCESS_RATE * len(successes), \
        f"successes {n}/{len(successes)}  mean return {mean:.1f}  gate >= {SUCCESS_RATE:.0%}"


def evaluate(model: BaseAlgorithm, name: str, episodes: int = 10, seed: int = 0) -> tuple[list[float], list[bool]]:
    env = make_env(name)

    def act(obs):
        return reference_action(model, obs)
    runs = [rollout(name, env, act, seed + i) for i in range(episodes)]
    return [r for r, _ in runs], [s for _, s in runs]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    name = sys.argv[1]
    check_env_name(name)
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else ENVS[name]["steps"]
    algo = ALGOS[ENVS[name].get("algo", "sac")]
    model = algo("MlpPolicy", make_env(name), policy_kwargs=dict(net_arch=[64, 64]), seed=0, verbose=0,
                 **ENVS[name].get("hp", {}))
    model.learn(total_timesteps=steps)
    OUT.mkdir(exist_ok=True)
    model.save(model_path(name))
    export_header(model, HEADER, name)
    ok, msg = gate(name, *evaluate(model, name))
    print(f"{name}  timesteps {steps}  {msg}  passed: {str(ok).lower()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
