/* Cart-pole physics, float32 port of cartpole.py (same constants and integrator; tests/test_cartpole_c.py
 * checks it against the Python env). Builds on the host (tests) and the ESP32. SPDX-License-Identifier: GPL-3.0-or-later */
#include "cartpole.h"

#include <math.h>

#define M_CART 1.0f
#define M_POLE 0.1f
#define HALF_LEN 0.5f
#define G 9.81f
#define DT (CARTPOLE_DT_MS / 1000.0f)
#define FORCE_MAX 10.0f
#define X_MAX 2.4f

int cartpole_step(float s[4], float action)
{
    float a = action > 1.0f ? 1.0f : action < -1.0f ? -1.0f : action;
    float u = FORCE_MAX * a;
    float sn = sinf(s[2]), c = cosf(s[2]), m = M_CART + M_POLE;
    float temp = (u + M_POLE * HALF_LEN * s[3] * s[3] * sn) / m;
    float th_acc = (G * sn - c * temp) / (HALF_LEN * (4.0f / 3.0f - M_POLE * c * c / m));
    float x_acc = temp - M_POLE * HALF_LEN * th_acc * c / m;
    /* Semi-implicit Euler: velocities first, positions with the new velocities. */
    s[1] += DT * x_acc;
    s[3] += DT * th_acc;
    s[0] += DT * s[1];
    s[2] += DT * s[3];
    return fabsf(s[0]) > X_MAX;
}

void cartpole_obs(const float s[4], float obs[CARTPOLE_OBS])
{
    obs[0] = s[0];
    obs[1] = s[1];
    obs[2] = cosf(s[2]);
    obs[3] = sinf(s[2]);
    obs[4] = s[3];
}
