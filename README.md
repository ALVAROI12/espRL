# espRL: RL on the ESP32 (hardware-in-the-loop)

A SAC policy for Gymnasium `Pendulum-v1` is trained on the Mac and runs as plain C (`mlp.c`) on an
Arduino Nano ESP32 (ESP32-S3). The Mac steps the pendulum and gets every action from the board over USB.
Design: `docs/superpowers/specs/2026-11-16-esp32-rl-hil-design.md`.

```
make setup               # .venv (torch, SB3, gymnasium)
make test                # host tests: C forward pass vs torch, serial protocol
make train               # ~25 s on an M-series Mac; writes out/sac_pendulum.zip + firmware/main/policy_weights.h
colima start             # Docker for ESP-IDF
make fw                  # firmware/build-esp32s3
make flash [PORT=...]    # default PORT = first /dev/cu.usbmodem*
make hil [PORT=...]      # 10 episodes on the board; report in out/hil_report.json
make viz [PORT=...]      # live window: pendulum + chip/torch torque, inference us, USB round trip
make viz VIZ_ARGS="--save out/hil.gif"   # record one episode instead
```

`make train` and `make fw` go together: the board must run the header from the same model `make hil` loads, or the
action-difference gate fails.

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

## Board notes

- The first flash over the Arduino bootloader fails once with "Device not configured": the board then
  re-enumerates as the ROM's USB-Serial/JTAG (303A:1001) and the next `make flash` works. Hold B1 to GND
  and reset if it never does. The Arduino sketch is gone after flashing; the Arduino IDE can restore it.
- `make flash` uses `--before usb-reset` and retries once: the default reset fails about every other time
  while the app runs.
- Only `main.c` and `mlp.c` are built `-O2`. With the whole image at `-O2`
  (`CONFIG_COMPILER_OPTIMIZATION_PERF`) the board answered one request per port open and then went silent;
  not traced further inside IDF.
