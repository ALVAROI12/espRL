"""Live view of the loop: the env the ESP32 is controlling (pendulum, cart-pole, lunar lander), plus action / compute /
timing traces.

Usage: python viz.py PORT [--env ENV] [--onchip] [--save out.gif] [--episodes N]
HIL (default): the Mac steps the env and asks the chip for each action; traces show inference us and USB round trip.
--onchip (cart-pole only): the chip runs physics and policy at 50 Hz and streams each step; traces show compute per
step and the chip's step period.
Without --save it opens a window and runs episodes until closed; with --save it records N episodes (default 1).
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from stable_baselines3 import SAC

import onchip
import train
from projects.cartpole.cartpole import HALF_LEN, OUT_OF_TRACK, X_MAX, obs_of, reward
from export import reference_action
from hil import CPU_MHZ, ready
from link import Link

# One sample per step: obs to draw, chip action(s), torch action(s), us (compute), ms (timing), reward, done, lost,
# img (env frame, lander only).


def hil_samples(link, model, env, render=False):
    obs, ep = env.reset(seed=0)[0], 0
    while True:
        ref = reference_action(model, obs)
        t0 = time.perf_counter()
        r = link.act(obs)
        ms = (time.perf_counter() - t0) * 1e3
        action, us = (ref, np.nan) if r is None else (np.array(r[0], dtype=np.float32), r[1] / CPU_MHZ)
        draw = obs
        obs, rew, term, trunc, _ = env.step(action)
        yield dict(obs=draw, action=action, ref=ref, us=us, ms=ms, reward=float(rew), done=term or trunc,
                   lost=int(r is None), img=env.render() if render else None)
        if term or trunc:
            ep += 1
            obs = env.reset(seed=ep)[0]


def onchip_samples(link, model, env):
    ep = 0
    while True:
        env.reset(seed=ep)
        prev, t_prev, n = np.float32(env.unwrapped.state), None, 0
        for step, state, action, cyc, t_us, flags in onchip.stream(link, prev):
            ms = np.nan if t_prev is None else ((t_us - t_prev) % 2**32) / 1e3
            yield dict(obs=obs_of(state), action=action, ref=reference_action(model, obs_of(prev))[0], us=cyc / CPU_MHZ,
                       ms=ms, reward=OUT_OF_TRACK if flags & 1 else reward(state, action), done=flags != 0,
                       lost=step - n)
            prev, t_prev, n = state, t_us, step + 1
            if flags:
                break
        else:  # stream ended without the end flag: a late frame
            yield dict(obs=obs_of(prev), action=np.nan, ref=np.nan, us=np.nan, ms=np.nan, reward=0.0, done=True, lost=1)
        ep += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--save")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--env", default="pendulum")
    ap.add_argument("--onchip", action="store_true")
    args = ap.parse_args()
    if args.onchip and not args.env.startswith("cartpole"):
        sys.exit("--onchip runs the cart-pole physics on the chip: use --env cartpole_balance or cartpole_swingup")
    if not ready(args.env):
        sys.exit(2)
    lander, cart = args.env == "lunarlander", args.env.startswith("cartpole")
    model = SAC.load(train.model_path(args.env))
    n_act = model.action_space.shape[0]
    link = Link(args.port, n_act=n_act)
    env = train.make_env(args.env, render_mode="rgb_array") if lander else train.make_env(args.env)
    steps = env.spec.max_episode_steps  # traces show one episode
    scale = float(model.action_space.high[0])
    samples = onchip_samples(link, model, env) if args.onchip else hil_samples(link, model, env, render=lander)

    if args.save:  # on-chip: record first (the chip does not wait for rendering), then draw every step
        if args.onchip:
            recorded, eps = [], 0
            while eps < args.episodes:
                recorded.append(next(samples))
                eps += recorded[-1]["done"]
            it = iter(recorded)
        else:
            it = samples

        def batch():  # FuncAnimation also draws an init frame, so the recording runs out one call early
            s = next(it, None)
            return [] if s is None else [s]
    elif args.onchip:  # live on-chip: a reader thread keeps up with the 50 Hz stream, each redraw takes what came in
        q = queue.Queue()
        threading.Thread(target=lambda: [q.put(s) for s in samples], daemon=True).start()

        def batch():
            out = []
            while not q.empty():
                out.append(q.get())
            return out
    else:
        def batch():
            return [next(samples)]

    st = {"ep": 0, "step": 0, "ret": 0.0, "lost": 0, "ended": False, "done_eps": 0}
    hist = {k: deque(maxlen=steps) for k in ("chip", "torch", "us", "ms")}

    fig = plt.figure(figsize=(11, 5.5))
    ax_p = fig.add_subplot(1, 2, 1)
    ax_a, ax_us, ax_ms = (fig.add_subplot(3, 2, i) for i in (2, 4, 6))
    if lander:  # the env's own rendering
        ax_p.set(xticks=[], yticks=[])
        img = ax_p.imshow(np.zeros((400, 600, 3), dtype=np.uint8))
    elif cart:  # track with end stops, cart, pole of length 2 * HALF_LEN from the cart's center
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
    names = ["main", "lateral"] if lander else [""]
    l_chip = [ax_a.plot([], [], label=f"ESP32 {n}".strip(), lw=2)[0] for n in names]
    l_torch = [ax_a.plot([], [], "k--", label="torch (Mac)" if i == 0 else None, lw=1)[0] for i in range(n_act)]
    l_us, = ax_us.plot([], [], color="#7d3c98")
    l_ms, = ax_ms.plot([], [], color="#117a65")
    ax_a.set(ylim=(-1.1 * scale, 1.1 * scale), xlim=(0, steps), ylabel="force / 10" if cart else "engines" if lander else "torque")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_us.set(xlim=(0, steps), ylabel="step compute us" if args.onchip else "inference us")
    ax_ms.set(xlim=(0, steps), ylabel="chip step period ms" if args.onchip else "USB rtt ms", xlabel="step")
    if args.onchip:
        ax_p.text(0.02, 0.96, "physics + policy on the ESP32, 50 Hz", transform=ax_p.transAxes, fontsize=8, va="top")
    fig.tight_layout()

    def update(_):
        got = batch()
        for s in got:
            if st["ended"]:
                st.update(ep=st["ep"] + 1, step=0, ret=0.0, ended=False)
            st["ret"] += s["reward"]
            st["step"] += 1
            st["lost"] += s["lost"]
            st["ended"] = s["done"]
            st["done_eps"] += s["done"]
            for k, v in (("chip", np.atleast_1d(s["action"])), ("torch", np.atleast_1d(s["ref"])), ("us", s["us"]),
                         ("ms", s["ms"])):
                hist[k].append(v)
        if not got:
            return ()
        obs, action = got[-1]["obs"], float(np.atleast_1d(got[-1]["action"])[0])
        if lander:
            img.set_data(got[-1]["img"])
            torque.set_visible(False)
        elif cart:  # obs = [x, x_dot, cos th, sin th, th_dot]; arrow = force on the cart, 0.5 m at full force
            x = float(obs[0])
            body.set_x(x - 0.2)
            rod.set_data([x, x + 2 * HALF_LEN * obs[3]], [0, 2 * HALF_LEN * obs[2]])
            torque.xyann = (x, 0)
            torque.xy = (x + 0.5 * np.nan_to_num(action), 0)
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
        chip, ref = np.array(hist["chip"]), np.array(hist["torch"])
        for i in range(n_act):
            l_chip[i].set_data(xs, chip[:, i])
            l_torch[i].set_data(xs, ref[:, i])
        l_us.set_data(xs, hist["us"])
        l_ms.set_data(xs, hist["ms"])
        for ax, k in ((ax_us, "us"), (ax_ms, "ms")):
            v = np.array(hist[k], dtype=float)
            if np.isfinite(v).any():
                ax.set_ylim(0, np.nanmax(v) * 1.3)
        return ()

    def frames():  # recording: until the requested episodes are done
        i = 0
        while st["done_eps"] < args.episodes:
            yield i
            i += 1

    fps = train.ENVS[args.env]["fps"]  # 1 / env dt: GIFs play in real time
    anim = FuncAnimation(fig, update, frames=frames if args.save else None, interval=1000 / fps,
                         save_count=args.episodes * steps + 1, blit=False, cache_frame_data=False)
    try:
        if args.save:
            anim.save(args.save, writer=PillowWriter(fps=fps), dpi=100)  # 1100 x 550
            print(f"saved {args.save}")
        else:
            plt.show()
    finally:
        link.close()


if __name__ == "__main__":
    main()
