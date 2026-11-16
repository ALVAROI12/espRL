# espRL: RL on the ESP32 (hardware-in-the-loop)

A SAC policy is trained on the Mac and runs as plain C (`mlp.c`) on an Arduino Nano ESP32 (ESP32-S3).
The Mac steps the env and gets every action from the board over USB. Envs: Gymnasium `Pendulum-v1`, and an own
continuous cart-pole (`cartpole.py`) with two tasks, balance and swing-up.
Design: `docs/superpowers/specs/2026-11-16-esp32-rl-hil-design.md`, `docs/superpowers/specs/2026-11-16-cartpole-design.md`.

```
make setup               # .venv (torch, SB3, gymnasium)
make test                # host tests: C forward pass vs torch, serial protocol
make train [ENV=...]     # pendulum (default) | cartpole_balance | cartpole_swingup;
                         # writes out/<env>.zip + firmware/main/policy_weights.h
colima start             # Docker for ESP-IDF
make fw                  # firmware/build-esp32s3
make flash [PORT=...]    # default PORT = first /dev/cu.usbmodem*
make hil [ENV=...]       # 10 episodes on the board; report in out/hil_<env>.json
make viz [ENV=...]       # live window: pendulum or cart + chip/torch action, inference us, USB round trip
make viz VIZ_ARGS="--save out/hil.gif"   # record one episode instead
make onchip ENV=cartpole_swingup         # cart-pole physics + policy both on the chip, 50 Hz; out/onchip_<env>.json
make viz ENV=cartpole_swingup VIZ_ARGS=--onchip   # live view of the on-chip run
```

`make train ENV=x` and `make fw flash` go together: the board must run the header from the same model `make hil`
loads. `hil.py`/`viz.py` refuse to run (exit 2) when the header was exported for another env. `PORT=` works on
every board target (default: first /dev/cu.usbmodem*).

## Results (2026-11-16, Arduino Nano ESP32, ESP32-S3 rev v0.2 at 240 MHz)

| Metric | Value | Gate |
|---|---|---|
| Steps / lost replies | 2000 / 0 | 0 lost |
| Mean return, chip (10 episodes) | -131.6 | > -250 |
| Mean return, Python alone, same seeds | -131.6 | reported |
| Max abs action difference chip vs torch | 1.85e-6 | < 1e-4 |
| Inference, 3-64-64-1 float MLP | 99.0 us p50, 99.3 us p99 (23.8k cycles) | reported |
| USB round trip | 0.22 ms p50, 0.29 ms p99 | reported |

Policy: SAC, 20k steps, seed 0. Inference is with `mlp.c` at `-O2`; at the IDF default `-Og` it was 394 us.

### Cart-pole (2026-11-16, same board)

Own env (`cartpole.py`): frictionless cart-pole, 1 kg cart, 0.1 kg pole of 1 m, force +-10 N, track +-2.4 m,
50 Hz, 500 steps. Success = never leaves the track and cos th > 0.95 for each of the last 100 steps.

| Metric | Balance | Swing-up | Gate |
|---|---|---|---|
| Steps / lost replies | 5000 / 0 | 5000 / 0 | 0 lost |
| Successes, chip (10 episodes) | 10/10 | 10/10 | >= 9/10 |
| Mean return, chip / Python alone | 497.8 / 497.8 | 391.2 / 391.2 | reported |
| Max abs action difference chip vs torch | 1.66e-6 | 3.07e-6 | < 1e-4 |
| Inference, 5-64-64-1 float MLP | 105.3 us p50, 105.6 us p99 | 105.3 us p50, 111.1 us p99 | reported |
| USB round trip | 0.23 ms p50, 0.32 ms p99 | 0.23 ms p50, 0.32 ms p99 | reported |

Policies: SAC, seed 0, 60k steps (balance; 30k missed the gate) and 200k steps (swing-up), about 1 and 4 min
on the Mac. `make viz ENV=cartpole_swingup VIZ_ARGS="--save out/cartpole_swingup.gif"` records the swing-up.

### Physics on the chip too (2026-11-16, same board)

`make onchip`: the ESP32 runs the cart-pole physics (`cartpole.c`, a float32 port of `cartpole.py`) and the policy,
one step every 20 ms (FreeRTOS `vTaskDelayUntil`), and streams every step over USB. The Mac only sends each
episode's initial state, then checks the stream: the torch actor on the chip's own observation (action diff),
successes on the chip's own trajectory, no dropped frames, and the step period. The chip computes in float32, so
on swing-up its trajectory is not compared with the Python float64 run; `tests/test_cartpole_c.py` checks the C
physics step by step against `cartpole.py` instead. The pendulum stays HIL only.

| Metric | Balance | Swing-up | Gate |
|---|---|---|---|
| Steps / incomplete episodes | 5000 / 0 | 5000 / 0 | 0 incomplete |
| Successes, chip (10 episodes) | 10/10 | 10/10 | >= 9/10 |
| Max abs action difference chip vs torch | 1.85e-6 | 3.23e-6 | < 1e-4 |
| Step period | 19997-20001 us | 19997-20000 us | 20000 +- 1000 us |
| Compute per step (obs + policy + physics) | 108.6 us p50 | 110.4 us p50 | reported |

## Clips

Recorded from the board with `make viz ... VIZ_ARGS="[--onchip] --save out/x.gif"` (re-encoded with ffmpeg).
On-chip clips play in real time (50 fps); HIL clips play at 20 fps.

Swing-up, physics and policy on the ESP32:

![cart-pole swing-up on the ESP32](media/cartpole_swingup_onchip.gif)

Balance, physics and policy on the ESP32:

![cart-pole balance on the ESP32](media/cartpole_balance_onchip.gif)

HIL (physics on the Mac, policy on the ESP32): [pendulum](media/pendulum_hil.gif),
[cart-pole balance](media/cartpole_balance_hil.gif), [cart-pole swing-up](media/cartpole_swingup_hil.gif).

## Board notes

- The first flash over the Arduino bootloader fails once with "Device not configured": the board then
  re-enumerates as the ROM's USB-Serial/JTAG (303A:1001) and the next `make flash` works. Hold B1 to GND
  and reset if it never does. The Arduino sketch is gone after flashing; the Arduino IDE can restore it.
- `make flash` uses `--before usb-reset` and retries once: the default reset fails about every other time
  while the app runs.
- Only `main.c` and `mlp.c` are built `-O2`. With the whole image at `-O2`
  (`CONFIG_COMPILER_OPTIMIZATION_PERF`) the board answered one request per port open and then went silent;
  not traced further inside IDF.
