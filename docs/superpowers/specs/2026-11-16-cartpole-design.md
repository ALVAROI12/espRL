# Cart-pole on the ESP32 (balance + swing-up) - design

Date: 2026-11-16. Status: approved by the user (both tasks; continuous force, own env).

## Intent

Add a cart-pole to the hardware-in-the-loop pipeline that already flies Pendulum-v1 on the Arduino Nano ESP32:
two tasks, balance and swing-up, both trained on the Mac with SAC and run as C on the board, with the live
visualizer drawing the cart. The pendulum keeps working.

## Goals

1. `cartpole.py`: a Gymnasium `Env` with continuous force and two modes, `balance` and `swingup`.
2. Train, export, flash and run either task (or the pendulum) through the existing `make` targets with `ENV=`.
3. Pass the gates below on the board for both cart-pole tasks.
4. `viz.py` draws a cart on a track for the cart-pole envs (pendulum view unchanged).

## Non-goals

- Gymnasium's discrete `CartPole-v1` (would need a second algorithm and output path on the chip).
- More than one policy in one firmware image (one policy per flash).
- Real actuators; on-device training (project B); quantization (project C).

## Constraints

- Everything in AGENTS.md (IDF in Docker, -O2 only for `main.c`/`mlp.c`, no allocation on the chip).
- The pendulum results in README.md must still reproduce with `ENV=pendulum`.

## Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| Own continuous cart-pole env, two start modes, SAC for both | One training path, one firmware output path; must write and test the physics | **Chosen** |
| Gymnasium `CartPole-v1` (discrete) for balance + own env for swing-up | Standard env, but PPO/DQN + argmax output on the chip: two pipelines | Rejected |
| MuJoCo `InvertedPendulum-v5` | Continuous, but balance only, adds the mujoco dependency | Rejected |

## Design

Physics (classic cart-pole, Florian 2007 corrected equations, frictionless), Euler or semi-implicit Euler at
dt = 0.02 s (50 Hz): cart mass 1.0 kg, pole mass 0.1 kg, pole half-length 0.5 m (1 m pole), g = 9.81,
force u in [-10, 10] N, track |x| <= 2.4 m, episode 500 steps (10 s).

Observation (5): `[x, x_dot, cos th, sin th, th_dot]`, th = 0 upright. Action (1): force / 10 in [-1, 1] mapped
to N (SAC's action space is `Box(-1, 1)`; the C output scale is the env's bound, as for the pendulum).

Modes:
- `balance`: start th ~ U(-0.05, 0.05) rad, x, x_dot, th_dot ~ U(-0.05, 0.05).
- `swingup`: start th = pi + U(-0.05, 0.05) (hanging), others ~ U(-0.05, 0.05).

Reward (same for both, per step, our call; the builder may tune the weights but not the success test):
`cos th - 0.1 * x^2 - 0.001 * u^2`; leaving the track terminates the episode with -100.

Success test (the gate, per episode): cart never leaves the track and cos th > 0.95 for the whole last 2 s
(100 steps). Gate: at least 9 of 10 episodes succeed on the board, plus the existing gates: chip vs torch
max |action diff| < 1e-4 in lockstep, 0 lost replies. For the pendulum the existing return gate stays.

Pipeline changes:
- `mlp.c`: `float mlp_forward(const float *obs)` for any `POLICY_OBS`; keep `POLICY_ACT == 1`.
- Protocol: request = `0xA5`, `seq`, `POLICY_OBS` x f32 (14 B for the pendulum, 22 B for the cart-pole);
  reply unchanged. Firmware reads `4 * POLICY_OBS` bytes.
- `make train ENV=pendulum|cartpole_balance|cartpole_swingup` writes `out/<env>.zip` and the header (header
  records the env name as `POLICY_ENV`); `make hil ENV=...` / `make viz ENV=...` load `out/<env>.zip` and refuse
  to run if the header's `POLICY_ENV` differs (a flashed header from another env).
- Tests: physics sanity (hanging pole at rest with u = 0 stays hanging; upright pole with a small tilt falls;
  force pushes the cart the right way), C vs torch for a cart-pole actor (5 obs), link with 5 obs.

## Left to the builder

- SAC timesteps per task (expect tens of thousands for balance, 100k-300k for swing-up) and reward weights.
- Integrator detail (Euler vs semi-implicit) as long as the sanity tests pass.
- Visualizer layout for the cart (track width, cart size), and README results table for the new tasks.
