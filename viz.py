"""Live view of the HIL loop: the pendulum the ESP32 is controlling, plus action / inference / round-trip traces.

Usage: python viz.py PORT [--save out.gif] [--episodes N]
Without --save it opens a window and runs episodes until closed; with --save it records N episodes (default 1).
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import SAC

from export import reference_action
from hil import CPU_MHZ
from link import Link
from train import MODEL

HISTORY = 200  # steps shown in the traces (one episode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--save")
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args()
    model, link, env = SAC.load(MODEL), Link(args.port), gym.make("Pendulum-v1")
    st = {"obs": env.reset(seed=0)[0], "ep": 0, "step": 0, "ret": 0.0, "lost": 0}
    hist = {k: deque(maxlen=HISTORY) for k in ("chip", "torch", "us", "ms")}

    fig = plt.figure(figsize=(11, 5.5))
    ax_p = fig.add_subplot(1, 2, 1)
    ax_a, ax_us, ax_ms = (fig.add_subplot(3, 2, i) for i in (2, 4, 6))
    ax_p.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal", xticks=[], yticks=[])
    rod, = ax_p.plot([], [], lw=8, color="#c0392b", solid_capstyle="round")
    ax_p.plot(0, 0, "ko", ms=8)
    torque = ax_p.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2, color="#2471a3"))
    title = ax_p.set_title("episode 0  step 0  return 0  lost 0")  # placeholder so tight_layout leaves room
    l_chip, = ax_a.plot([], [], label="ESP32", lw=2)
    l_torch, = ax_a.plot([], [], "--", label="torch (Mac)", lw=1)
    l_us, = ax_us.plot([], [], color="#7d3c98")
    l_ms, = ax_ms.plot([], [], color="#117a65")
    ax_a.set(ylim=(-2.2, 2.2), xlim=(0, HISTORY), ylabel="torque")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_us.set(xlim=(0, HISTORY), ylabel="inference us")
    ax_ms.set(xlim=(0, HISTORY), ylabel="USB rtt ms", xlabel="step")
    fig.tight_layout()

    def update(_):
        obs = st["obs"]
        ref = reference_action(model, obs)
        t0 = time.perf_counter()
        r = link.act(obs)
        ms = (time.perf_counter() - t0) * 1e3
        if r is None:
            st["lost"] += 1
            action, us = ref, np.nan
        else:
            action, us = r[0], r[1] / CPU_MHZ
        st["obs"], reward, term, trunc, _ = env.step(np.array([action], dtype=np.float32))
        st["ret"] += float(reward)
        st["step"] += 1
        for k, v in (("chip", action), ("torch", ref), ("us", us), ("ms", ms)):
            hist[k].append(v)

        th = np.arctan2(obs[1], obs[0])  # 0 = upright
        x, y = np.sin(th), np.cos(th)
        rod.set_data([0, x], [0, y])
        # Arrow at the tip along the direction a positive torque turns it (d(sin th, cos th)/d th), length ~ |torque|.
        k = 0.25 * action
        torque.xyann = (x, y)
        torque.xy = (x + k * np.cos(th), y - k * np.sin(th))
        title.set_text(f"episode {st['ep']}  step {st['step']}  return {st['ret']:.0f}  lost {st['lost']}")
        xs = range(len(hist["chip"]))
        l_chip.set_data(xs, hist["chip"])
        l_torch.set_data(xs, hist["torch"])
        l_us.set_data(xs, hist["us"])
        l_ms.set_data(xs, hist["ms"])
        for ax, k in ((ax_us, "us"), (ax_ms, "ms")):
            v = np.array(hist[k], dtype=float)
            if np.isfinite(v).any():
                ax.set_ylim(0, np.nanmax(v) * 1.3)
        if term or trunc:
            st["ep"] += 1
            st["step"], st["ret"] = 0, 0.0
            st["obs"] = env.reset(seed=st["ep"])[0]
        return rod, torque, title, l_chip, l_torch, l_us, l_ms

    anim = FuncAnimation(fig, update, frames=args.episodes * 200 if args.save else None, interval=50,
                         blit=False, cache_frame_data=False)
    try:
        if args.save:
            anim.save(args.save, writer=PillowWriter(fps=20))
            print(f"saved {args.save}")
        else:
            plt.show()
    finally:
        link.close()


if __name__ == "__main__":
    main()
