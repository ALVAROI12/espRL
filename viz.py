"""Live view of the HIL loop: the pendulum or cart-pole the ESP32 is controlling, plus action / inference /
round-trip traces.

Usage: python viz.py PORT [--env ENV] [--save out.gif] [--episodes N]
Without --save it opens a window and runs episodes until closed; with --save it records N episodes (default 1).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from stable_baselines3 import SAC

from export import reference_action
import train
from cartpole import HALF_LEN, X_MAX
from hil import CPU_MHZ, ready
from link import Link


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--save")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--env", default="pendulum")
    args = ap.parse_args()
    if not ready(args.env):
        sys.exit(2)
    model, link, env = SAC.load(train.model_path(args.env)), Link(args.port), train.make_env(args.env)
    cart = args.env != "pendulum"
    steps = env.spec.max_episode_steps  # traces show one episode
    scale = float(model.action_space.high[0])
    st = {"obs": env.reset(seed=0)[0], "ep": 0, "step": 0, "ret": 0.0, "lost": 0}
    hist = {k: deque(maxlen=steps) for k in ("chip", "torch", "us", "ms")}

    fig = plt.figure(figsize=(11, 5.5))
    ax_p = fig.add_subplot(1, 2, 1)
    ax_a, ax_us, ax_ms = (fig.add_subplot(3, 2, i) for i in (2, 4, 6))
    if cart:  # track with end stops, cart, pole of length 2 * HALF_LEN from the cart's center
        ax_p.set(xlim=(-X_MAX - 0.3, X_MAX + 0.3), ylim=(-1.2, 1.2), aspect="equal", yticks=[], xlabel="x (m)")
        ax_p.plot([-X_MAX, X_MAX], [-0.1, -0.1], "k-", lw=1)
        ax_p.plot([-X_MAX] * 2 + [None] + [X_MAX] * 2, [-0.2, 0.1, None, -0.2, 0.1], "k-", lw=3)
        body = ax_p.add_patch(Rectangle((-0.2, -0.1), 0.4, 0.2, color="#555555"))
        rod, = ax_p.plot([], [], lw=6, color="#c0392b", solid_capstyle="round")
    else:
        ax_p.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal", xticks=[], yticks=[])
        rod, = ax_p.plot([], [], lw=8, color="#c0392b", solid_capstyle="round")
        ax_p.plot(0, 0, "ko", ms=8)
    torque = ax_p.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2, color="#2471a3"))
    title = ax_p.set_title("episode 0  step 0  return 0  lost 0")  # placeholder so tight_layout leaves room
    l_chip, = ax_a.plot([], [], label="ESP32", lw=2)
    l_torch, = ax_a.plot([], [], "--", label="torch (Mac)", lw=1)
    l_us, = ax_us.plot([], [], color="#7d3c98")
    l_ms, = ax_ms.plot([], [], color="#117a65")
    ax_a.set(ylim=(-1.1 * scale, 1.1 * scale), xlim=(0, steps), ylabel="force / 10" if cart else "torque")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_us.set(xlim=(0, steps), ylabel="inference us")
    ax_ms.set(xlim=(0, steps), ylabel="USB rtt ms", xlabel="step")
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

        if cart:  # obs = [x, x_dot, cos th, sin th, th_dot]; arrow = force on the cart, 0.5 m at full force
            x = float(obs[0])
            body.set_x(x - 0.2)
            rod.set_data([x, x + 2 * HALF_LEN * obs[3]], [0, 2 * HALF_LEN * obs[2]])
            torque.xyann = (x, 0)
            torque.xy = (x + 0.5 * action, 0)
        else:
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

    anim = FuncAnimation(fig, update, frames=args.episodes * steps if args.save else None, interval=50,
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
