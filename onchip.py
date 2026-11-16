"""Physics and policy both on the ESP32: the chip runs each cart-pole episode in real time (one step every 20 ms)
and streams every step; the Mac only sends the initial state, then checks the run.

Usage: python onchip.py PORT ENV [episodes]   ENV: cartpole_balance | cartpole_swingup
Exit 1 unless every gate passes (2 if not ready to run):
- successes on the chip's own trajectory, as in train.py (>= 90%);
- max |chip action - torch action on the chip's own observation| < 1e-4;
- every episode arrives whole (no dropped frame);
- every step starts within 1 ms of 20 ms after the previous one.
The chip computes in float32, so on swing-up its trajectory drifts from the Python (float64) run of the same seed:
that run is reported, not compared. projects/cartpole/test_cartpole_c.py checks the C physics step by step instead.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from stable_baselines3 import SAC

import train
from projects.cartpole.cartpole import OUT_OF_TRACK, obs_of, reward
from export import reference_action
from hil import CPU_MHZ, MAX_ACTION_DIFF, ready
from link import Link

PERIOD_US = 20_000
MAX_PERIOD_ERROR_US = 1_000
FRAME_TIMEOUT = 0.1  # s; a frame is due every 20 ms


def stream(link: Link, state: np.ndarray):
    """Start one on-chip episode from state; yield its frames as they arrive, until the end flag or a late frame."""
    seq = link.start_episode(state)
    while seq is not None:
        f = link.frame(seq, FRAME_TIMEOUT)
        if f is None:
            return
        yield f
        if f[5]:
            return


def episode(link: Link, state: np.ndarray) -> list[tuple]:
    return list(stream(link, state))


def complete(frames: list[tuple]) -> bool:
    return bool(frames) and [f[0] for f in frames] == list(range(len(frames))) and frames[-1][5] != 0


def run(link: Link, model: SAC, name: str, episodes: int = 10, seed: int = 0) -> dict:
    env = train.make_env(name)
    returns, successes, diffs, cycles, periods, incomplete = [], [], [], [], [], 0
    for i in range(episodes):
        env.reset(seed=seed + i)
        prev = np.float32(env.unwrapped.state)
        frames = episode(link, prev)
        incomplete += not complete(frames)
        total, cos_hist, terminated = 0.0, [], False
        for _, state, action, cyc, _, flags in frames:
            diffs.append(abs(action - reference_action(model, obs_of(prev))[0]))
            cycles.append(cyc)
            terminated = bool(flags & 1)
            total += OUT_OF_TRACK if terminated else reward(state, action)
            cos_hist.append(float(np.cos(state[2])))
            prev = state
        t = np.array([f[4] for f in frames], dtype=np.int64)
        periods += list((np.diff(t) % 2**32))
        returns.append(total)
        successes.append(train.success(cos_hist, terminated))
    py_returns, py_successes = train.evaluate(model, name, episodes, seed)
    ok, msg = train.gate(name, returns, successes)
    us = np.array(cycles, dtype=float) / CPU_MHZ if cycles else np.array([np.nan])
    p = np.array(periods, dtype=float) if periods else np.array([PERIOD_US], dtype=float)
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
        "max_period_error_us": float(np.abs(p - PERIOD_US).max()),
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
    if name == "pendulum":
        print("the pendulum's physics runs on the Mac only: use `make hil`", file=sys.stderr)
        return 2
    if not ready(name):
        return 2
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    model = SAC.load(train.model_path(name))
    link = Link(port)
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
    print(f"step period us {rep['period_us_min']:.0f}..{rep['period_us_max']:.0f}  gate {PERIOD_US} +- {MAX_PERIOD_ERROR_US}")
    print(f"compute per step (obs + policy + physics) us p50 {rep['step_compute_us_p50']:.1f}  "
          f"p99 {rep['step_compute_us_p99']:.1f}")
    print(f"passed: {str(rep['passed']).lower()}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
