"""Physics and policy both on the ESP32: the chip runs each episode in real time (one step per env dt: 20 ms for
the cart-pole, 200 ms for the acrobot) and streams every step; the Mac only sends the initial state, then checks.

Usage: python onchip.py PORT ENV [episodes]   ENV: an env with `sim` in train.ENVS (cartpole_*, acrobot)
Exit 1 unless every gate passes (2 if not ready to run):
- the env's gate (train.gate) on the chip's own trajectories;
- max |chip action - torch action on the chip's own observation| < 1e-4;
- every episode arrives whole (no dropped frame);
- every step starts within 1 ms of one env dt after the previous one.
The chip computes in float32, so on chaotic tasks its trajectory drifts from the Python (float64) run of the same
seed: that run is reported, not compared. Each project's test_*_c.py checks its C physics step by step instead.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

import train
from export import reference_action
from hil import CPU_MHZ, MAX_ACTION_DIFF, ready
from link import Link

MAX_PERIOD_ERROR_US = 1_000


def period_us(name: str) -> float:
    return 1e6 / train.ENVS[name]["fps"]


def stream(link: Link, state: np.ndarray, period_s: float = 0.02):
    """Start one on-chip episode from state; yield its frames as they arrive, until the end flag or a late frame."""
    seq = link.start_episode(state)
    while seq is not None:
        f = link.frame(seq, 2 * period_s + 0.05)
        if f is None:
            return
        yield f
        if f[5]:
            return


def episode(link: Link, state: np.ndarray, period_s: float = 0.02) -> list[tuple]:
    return list(stream(link, state, period_s))


def complete(frames: list[tuple]) -> bool:
    return bool(frames) and [f[0] for f in frames] == list(range(len(frames))) and frames[-1][5] != 0


def run(link: Link, model: BaseAlgorithm, name: str, episodes: int = 10, seed: int = 0) -> dict:
    env, sim, cos = train.make_env(name), train.ENVS[name]["sim"], train.ENVS[name].get("cos")
    period = period_us(name)
    returns, successes, diffs, cycles, periods, incomplete = [], [], [], [], [], 0
    for i in range(episodes):
        env.reset(seed=seed + i)
        prev = np.float32(env.unwrapped.state)
        frames = episode(link, prev, period / 1e6)
        incomplete += not complete(frames)
        total, cos_hist, terminated = 0.0, [], False
        for _, state, action, cyc, _, flags in frames:
            obs = sim.obs_of(prev)
            diffs.append(abs(action - reference_action(model, obs)[0]))
            cycles.append(cyc)
            terminated = bool(flags & 1)
            total += sim.step_reward(state, action, terminated)
            if cos is not None:
                cos_hist.append(float(sim.obs_of(state)[cos]))
            prev = state
        t = np.array([f[4] for f in frames], dtype=np.int64)
        periods += list((np.diff(t) % 2**32))
        returns.append(total)
        successes.append(train.success(cos_hist, terminated) if cos is not None else terminated)
    py_returns, py_successes = train.evaluate(model, name, episodes, seed)
    ok, msg = train.gate(name, returns, successes)
    us = np.array(cycles, dtype=float) / CPU_MHZ if cycles else np.array([np.nan])
    p = np.array(periods, dtype=float) if periods else np.array([period], dtype=float)
    rep = {
        "env": name,
        "returns_chip": returns,
        "successes_chip": successes,
        "returns_python": py_returns,
        "successes_python": py_successes,
        "gate_chip": msg,
        "max_abs_action_diff": float(max(diffs)) if diffs else float("nan"),
        "step_compute_us_p50": float(np.percentile(us, 50)),
        "step_compute_us_p99": float(np.percentile(us, 99)),
        "period_us_min": float(p.min()),
        "period_us_max": float(p.max()),
        "period_us": period,
        "max_period_error_us": float(np.abs(p - period).max()),
        "incomplete": incomplete,
        "steps": len(cycles),
    }
    rep["passed"] = bool(ok and incomplete == 0 and rep["max_abs_action_diff"] < MAX_ACTION_DIFF
                         and rep["max_period_error_us"] <= MAX_PERIOD_ERROR_US)
    return rep


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: onchip.py PORT ENV [episodes]", file=sys.stderr)
        return 2
    port, name = sys.argv[1], sys.argv[2]
    if not ready(name):
        return 2
    if "sim" not in train.ENVS[name]:
        print(f"{name}'s physics runs on the Mac only: use `make hil ENV={name}`", file=sys.stderr)
        return 2
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    model = train.load(name)
    link = Link(port)  # on-chip envs return 1 action
    try:
        rep = run(link, model, name, episodes)
    finally:
        link.close()
    (train.OUT / f"onchip_{name}.json").write_text(json.dumps(rep, indent=2))
    _, py_msg = train.gate(name, rep["returns_python"], rep["successes_python"])
    print(f"{name} on chip  steps {rep['steps']}  incomplete episodes {rep['incomplete']}")
    print(f"chip: {rep['gate_chip']}")
    print(f"python (float64, independent): {py_msg}")
    print(f"max |action chip - torch| {rep['max_abs_action_diff']:.2e}  gate < {MAX_ACTION_DIFF:.0e}")
    print(f"step period us {rep['period_us_min']:.0f}..{rep['period_us_max']:.0f}  gate {rep['period_us']:.0f} +- {MAX_PERIOD_ERROR_US}")
    print(f"compute per step (obs + policy + physics) us p50 {rep['step_compute_us_p50']:.1f}  "
          f"p99 {rep['step_compute_us_p99']:.1f}")
    print(f"passed: {str(rep['passed']).lower()}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
