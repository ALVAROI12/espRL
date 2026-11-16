"""Hardware-in-the-loop: Pendulum-v1 on the Mac, actions from the policy running on the ESP32.

Usage: python hil.py PORT [episodes]   (exit 1 unless every gate in the design passes)
Lockstep pass (gated): the torch actor sees the same observation as the chip at every step.
Independent pass (reported only): the torch actor flies the same seeds alone; swing-up is chaotic, so its
returns may differ from the chip's even when the actions match to 1e-6.
"""

from __future__ import annotations

import json
import sys
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

from export import reference_action
from link import Link
from train import MIN_RETURN, MODEL, evaluate

CPU_MHZ = 240
MAX_ACTION_DIFF = 1e-4


def run(link: Link, model: SAC, episodes: int = 10, seed: int = 0) -> dict:
    env = gym.make("Pendulum-v1")
    returns, diffs, cycles, rtts, lost, steps = [], [], [], [], 0, 0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, done = 0.0, False
        while not done:
            ref = reference_action(model, obs)
            t0 = time.perf_counter()
            r = link.act(obs)
            rtts.append((time.perf_counter() - t0) * 1e3)
            steps += 1
            if r is None:
                lost += 1
                action = ref  # keep the episode going; the lost gate already fails the run
            else:
                action, cyc = r
                diffs.append(abs(action - ref))
                cycles.append(cyc)
            obs, reward, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    us = np.array(cycles, dtype=float) / CPU_MHZ if cycles else np.array([np.nan])
    rep = {
        "returns_chip": returns,
        "returns_python": evaluate(model, episodes, seed),
        "mean_return_chip": float(np.mean(returns)),
        "max_abs_action_diff": float(max(diffs)) if diffs else float("nan"),
        "inference_us_p50": float(np.percentile(us, 50)),
        "inference_us_p99": float(np.percentile(us, 99)),
        "rtt_ms_p50": float(np.percentile(rtts, 50)),
        "rtt_ms_p99": float(np.percentile(rtts, 99)),
        "lost": lost,
        "steps": steps,
    }
    rep["passed"] = bool(lost == 0 and rep["max_abs_action_diff"] < MAX_ACTION_DIFF
                         and rep["mean_return_chip"] > MIN_RETURN)
    return rep


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: hil.py PORT [episodes]", file=sys.stderr)
        return 2
    if not MODEL.exists():
        print(f"{MODEL} missing: run `make rl-train` (it also rewrites firmware/main/policy_weights.h)", file=sys.stderr)
        return 2
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    model = SAC.load(MODEL)
    link = Link(sys.argv[1])
    try:
        rep = run(link, model, episodes)
    finally:
        link.close()
    (MODEL.parent / "hil_report.json").write_text(json.dumps(rep, indent=2))
    py = float(np.mean(rep["returns_python"]))
    print(f"steps {rep['steps']}  lost {rep['lost']}")
    print(f"mean return  chip {rep['mean_return_chip']:.1f}  python (independent) {py:.1f}  gate > {MIN_RETURN}")
    print(f"max |action chip - torch| {rep['max_abs_action_diff']:.2e}  gate < {MAX_ACTION_DIFF:.0e}")
    print(f"inference us p50 {rep['inference_us_p50']:.1f}  p99 {rep['inference_us_p99']:.1f}")
    print(f"round trip ms p50 {rep['rtt_ms_p50']:.2f}  p99 {rep['rtt_ms_p99']:.2f}")
    print(f"passed: {str(rep['passed']).lower()}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
