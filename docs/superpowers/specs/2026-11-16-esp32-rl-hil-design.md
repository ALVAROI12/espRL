# ESP32 RL hardware-in-the-loop (project A) - design

Date: 2026-11-16. Status: draft for approval.

## Intent

Start RL work on real ESP32 silicon before the drone exists. The board on hand is an Arduino Nano ESP32
(ESP32-S3 QFN56 rev v0.2, 2 x 240 MHz, 8 MB PSRAM, USB-Serial/JTAG on `/dev/cu.usbmodem3101`; no sensors,
no actuators). The drone is parked for now; this is a standalone RL track.

Project A proves the full loop "train on the Mac -> run the policy on the ESP32 -> close the loop against a
simulated plant" and measures what it costs on the chip. It is the base for project B (on-device training,
TinyRL) and later for policies on the flight controller: B and later projects reuse the export, serial link,
and timing harness built here.

## Goals

1. Train a continuous-control policy for pendulum swing-up (Gymnasium `Pendulum-v1`) on the Mac.
2. Export the actor to a C header; run it on the ESP32-S3 with a plain-C MLP forward pass.
3. Hardware-in-the-loop: the Mac steps the environment, sends the observation over USB serial, the ESP32
   returns the action; full episodes run this way.
4. Report: episode returns (chip vs. same policy in Python), max action difference, on-chip inference time
   (p50/p99, from CPU cycle counts), and serial round-trip time.

Success:
- Gate (lockstep): at every HIL step the torch actor is fed the same observation the chip got; max |chip action
  - torch action| < 1e-4 over all steps. (Our call: a second independent pass is chaotic for swing-up, so a
  1e-5 action difference can diverge the trajectory; its return difference is reported, not gated.)
- The policy swings up on the chip: mean HIL return over 10 episodes > -250 (typical trained SAC on
  Pendulum-v1: about -150).
- 10 HIL episodes (2000 steps) with zero lost or out-of-order packets.

## Non-goals

- On-device training (project B), quantization / ESP-DL / int8 (project C), real sensors or actuators.
- Any change to the drone firmware (`firmware/`), sim (`sim/`), or hardware (`hardware/`).

## Anti-goals

- A chip/Mac mismatch that goes unnoticed: every HIL run compares against the Python policy step by step.
- Heavy dependencies leaking into the existing `sim/.venv` (torch stays in its own venv).

## Constraints

- ESP-IDF `release-v5.5` in Docker via `tools/idf.sh` (Colima must be running: `colima start`).
- Build into `rl/firmware/build-esp32s3/` with `-D SDKCONFIG=build-esp32s3/sdkconfig`; never commit `sdkconfig`
  (an `sdkconfig.defaults` is committed).
- USB: the Arduino sketch enumerates as Arduino 2341:0070 (TinyUSB). Running `esptool` on it reset the chip into the
  ROM's USB-Serial/JTAG (303A:1001, `/dev/cu.usbmodem3101`), and `chip-id` worked there (2026-11-16). If the
  first flash fails to enter download mode, hold B1 to GND and reset. After the IDF app boots, the port stays
  USB-Serial/JTAG; `rl-flash`/`rl-hil` take `PORT=` and default to the first `/dev/cu.usbmodem*`.
- Flash from the Mac with `uvx esptool` (USB is not passed into the container). Flashing overwrites the Arduino
  sketch on the board; the user approved this (restorable from the Arduino IDE).
- Python 3.12 via uv, matching the repo; GPL-3.0.

## Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| A. Train on Mac, policy on ESP32, sim on Mac (HIL) | Easiest to debug; builds the link/export/timing reused later; no learning on chip | **Chosen by the user** |
| B. Train fully on ESP32 (RLtools, TinyRL) | Novel result; harder to debug; memory-bound | Next project, reuses A |
| C. Inference benchmark only (C / ESP-DL / RLtools, float vs int8) | Useful table, no RL behaviour | Falls out of A partly (inference timing) |

Within A, training library: **Stable-Baselines3 SAC** (Python) over RLtools. SB3 is the standard, easiest to
debug; the actor is a small torch MLP that exports trivially. RLtools comes in with project B, where on-chip
training needs it. On-chip inference: **plain C** (about 30 lines), not ESP-DL: a 3-64-64-1 float MLP needs no
library, and the identical source compiles on the host for the equivalence test.

## Design

Layout (all new, under `rl/`):

```
rl/
  requirements.txt     gymnasium, stable-baselines3, torch, numpy, pyserial
  train.py             SAC on Pendulum-v1 -> rl/out/sac_pendulum.zip; exports firmware/main/policy_weights.h
  export.py            actor (deterministic: tanh mean, scaled to [-2, 2]) -> C arrays (float32)
  hil.py               runs N episodes: env on Mac, action from ESP32; same seeds with the Python actor;
                       writes rl/out/hil_report.json + prints a summary table
  mlp.c / mlp.h        forward pass, static buffers, no allocation; shared by firmware and host test
  tests/test_mlp.py    builds mlp.c with cc into a shared lib, ctypes, compares to the torch actor
                       on 1000 random observations (max abs error < 1e-5); host only, no board
  firmware/            ESP-IDF app (target esp32s3)
    CMakeLists.txt, sdkconfig.defaults
    main/main.c        USB-Serial/JTAG loop: read request, run mlp, reply
    main/policy_weights.h   generated by export.py, committed so the firmware builds without training
```

Policy: SB3 SAC default actor `net_arch=[64, 64]`, ReLU. Deterministic action = `tanh(mu) * 2.0`
(Pendulum action bound). Observation `[cos th, sin th, th_dot]`, passed unnormalized (SB3 default for SAC).

Serial protocol (little-endian, fixed size, on the USB-Serial/JTAG port):
- Request (Mac -> ESP32), 14 bytes: `0xA5`, `seq` u8, 3 x float32 observation.
- Reply (ESP32 -> Mac), 10 bytes: `0x5A`, `seq` u8 (echo), float32 action, u32 inference CPU cycles.
- The ESP32 resyncs by scanning for `0xA5`; the Mac checks `seq` and a 100 ms reply timeout (counts as lost).

Firmware: `usb_serial_jtag` driver; console output and boot logs moved off the data channel
(`CONFIG_ESP_CONSOLE_NONE` or UART0) so only protocol bytes cross USB. Inference timed with
`esp_cpu_get_cycle_count()` around the forward pass; 240 MHz CPU clock set in `sdkconfig.defaults`.

Reference for comparisons is the torch actor everywhere: host test (C vs torch, < 1e-5) isolates export and C
math bugs; HIL (chip vs torch, < 1e-4) adds compiler/FPU effects (FMA contraction on the S3).

Repo housekeeping (our call): `.gitignore` gets `rl/.venv/`, `rl/out/`, `rl/firmware/build-*`,
`rl/firmware/sdkconfig`; `rl/out/` artifacts (model zip, `hil_report.json`) stay uncommitted.

Make targets: `rl-setup` (uv venv `rl/.venv`), `rl-train`, `rl-test` (host test), `rl-fw` (build via
`tools/idf.sh rl/firmware`), `rl-flash PORT=...` (`uvx esptool` using the build's `flash_args`), `rl-hil PORT=...`.

## Left to the builder

- Exact SAC hyperparameters and timesteps (SB3 defaults; about 20k-50k steps is typical for Pendulum).
- `hil.py` runs the lockstep comparison (gated) and a second independent pass with the same seeds (reported).
- Report format details beyond the fields listed in Goals.
