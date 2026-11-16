"""Hardware-in-the-loop: the env steps on the Mac, every action comes from the policy running on the ESP32.

Usage: python hil.py PORT ENV [episodes]   (exit 1 unless every gate passes; 2 if not ready to run)
Lockstep pass (gated): the torch actor sees the same observation as the chip at every step.
Independent pass (reported only): the torch actor runs the same seeds alone; swing-ups are chaotic, so its
returns may differ from the chip's even when the actions match to 1e-6.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from stable_baselines3 import SAC

import train
from export import header_env, reference_action
from link import Link

CPU_MHZ = 240
MAX_ACTION_DIFF = 1e-4


def ready(name: str) -> bool:
    """The trained model exists and the header (what `make fw` flashes) was exported for the same env."""
    train.check_env_name(name)
    if not train.model_path(name).exists():
        print(f"{train.model_path(name)} missing: run `make train ENV={name}`, then `make fw flash`", file=sys.stderr)
        return False
    flashed = header_env(train.HEADER)
    if flashed != name:
        print(f"{train.HEADER} is for {flashed!r}, not {name!r}: run `make train ENV={name}`, then `make fw flash`",
              file=sys.stderr)
        return False
    return True


def run(link: Link, model: SAC, name: str, episodes: int = 10, seed: int = 0) -> dict:
    env = train.make_env(name)
    diffs, cycles, rtts, lost = [], [], [], [0]

    def act(obs):
        ref = reference_action(model, obs)
        t0 = time.perf_counter()
        r = link.act(obs)
        rtts.append((time.perf_counter() - t0) * 1e3)
        if r is None:
            lost[0] += 1
            return ref  # keep the episode going; the lost gate already fails the run
        diffs.append(abs(r[0] - ref))
        cycles.append(r[1])
        return r[0]

    runs = [train.rollout(name, env, act, seed + i) for i in range(episodes)]
    returns, successes = [r for r, _ in runs], [s for _, s in runs]
    py_returns, py_successes = train.evaluate(model, name, episodes, seed)
    ok, msg = train.gate(name, returns, successes)
    us = np.array(cycles, dtype=float) / CPU_MHZ if cycles else np.array([np.nan])
    rep = {
        "env": name,
        "returns_chip": returns,
        "successes_chip": successes,
        "returns_python": py_returns,
        "successes_python": py_successes,
        "gate_chip": msg,
        "mean_return_chip": float(np.mean(returns)),
        "max_abs_action_diff": float(max(diffs)) if diffs else float("nan"),
        "inference_us_p50": float(np.percentile(us, 50)),
        "inference_us_p99": float(np.percentile(us, 99)),
        "rtt_ms_p50": float(np.percentile(rtts, 50)),
        "rtt_ms_p99": float(np.percentile(rtts, 99)),
        "lost": lost[0],
        "steps": len(rtts),
    }
    rep["passed"] = bool(lost[0] == 0 and rep["max_abs_action_diff"] < MAX_ACTION_DIFF and ok)
    return rep


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: hil.py PORT ENV [episodes]", file=sys.stderr)
        return 2
    port, name = sys.argv[1], sys.argv[2]
    if not ready(name):
        return 2
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    model = SAC.load(train.model_path(name))
    link = Link(port)
    try:
        rep = run(link, model, name, episodes)
    finally:
        link.close()
    (train.OUT / f"hil_{name}.json").write_text(json.dumps(rep, indent=2))
    _, py_msg = train.gate(name, rep["returns_python"], rep["successes_python"])
    print(f"{name}  steps {rep['steps']}  lost {rep['lost']}")
    print(f"chip: {rep['gate_chip']}")
    print(f"python (independent): {py_msg}")
    print(f"max |action chip - torch| {rep['max_abs_action_diff']:.2e}  gate < {MAX_ACTION_DIFF:.0e}")
    print(f"inference us p50 {rep['inference_us_p50']:.1f}  p99 {rep['inference_us_p99']:.1f}")
    print(f"round trip ms p50 {rep['rtt_ms_p50']:.2f}  p99 {rep['rtt_ms_p99']:.2f}")
    print(f"passed: {str(rep['passed']).lower()}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
