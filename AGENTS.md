# espRL - agent guide

RL policies trained on the Mac and run as plain C on an ESP32-S3 (Arduino Nano ESP32), closed-loop over
USB (hardware-in-the-loop). Split out of the ESP-FC Next drone repo (`~/Projects/espdrone`, `rl/`) on 2026-11-16.

## Rules
- ESP-IDF `release-v5.5` runs only in Docker via `tools/idf.sh` (Colima on macOS: `colima start`).
- Firmware builds into `firmware/build-esp32s3/` with `-D SDKCONFIG=build-esp32s3/sdkconfig`; never commit `sdkconfig`.
- On-chip code (`mlp.c`, `projects/*/*.c`, `firmware/`): no dynamic allocation; all of it except `firmware/` must build on host (tests) and target.
- Every policy change is gated on the board: `make hil` (lockstep chip vs torch action diff, return, lost replies).
- Only `main.c` and `mlp.c` are built `-O2`: the whole image at `-O2` broke the USB-Serial/JTAG link (README).
- One folder per project: `projects/<name>/` holds its env code, C physics port, tests, README (results) and
  `media/` clips. Shared pipeline stays at the root. A new env starts as a new folder.
- License: GPL-3.0.

## Commands
- `make setup` / `make test` - uv venv (Python 3.12, torch, SB3, gymnasium) and host tests
- `make train` - train + export `firmware/main/policy_weights.h` (fails under the return gate)
- `make fw` / `make flash [PORT=...]` - build in the IDF container, flash from the host with `uvx esptool`
- `make hil` / `make viz` - run episodes on the board; live visualizer (`VIZ_ARGS="--save out/hil.gif"` records)
- `ENV=pendulum|cartpole_balance|cartpole_swingup|lunarlander|acrobot` on train/hil/onchip/viz; the flashed header must match `ENV`
- `make onchip ENV=cartpole_*|acrobot` - physics + policy on the chip in real time (`VIZ_ARGS=--onchip` to watch);
  a new on-chip env needs a C port in its project folder, a `SIMS` row in `firmware/main/main.c` and `sim=` in `train.ENVS`
