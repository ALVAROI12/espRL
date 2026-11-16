/* Acrobot physics, float32 port of gymnasium/envs/classic_control/acrobot.py (same constants, RK4 step, angle
 * wrap and velocity bounds; test_acrobot_c.py checks it against Gymnasium). Builds on the host and the ESP32.
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "acrobot.h"

#include <math.h>

#define PI_F 3.14159265358979f
#define L1 1.0f   /* link 1 length */
#define M1 1.0f   /* link masses */
#define M2 1.0f
#define LC1 0.5f  /* centers of mass */
#define LC2 0.5f
#define I1 1.0f   /* moments of inertia */
#define I2 1.0f
#define G 9.8f
#define DT (ACROBOT_DT_MS / 1000.0f)
#define MAX_VEL_1 (4.0f * PI_F)
#define MAX_VEL_2 (9.0f * PI_F)

static void dsdt(const float s[4], float a, float d[4])
{
    float th1 = s[0], th2 = s[1], dth1 = s[2], dth2 = s[3];
    float d1 = M1 * LC1 * LC1 + M2 * (L1 * L1 + LC2 * LC2 + 2.0f * L1 * LC2 * cosf(th2)) + I1 + I2;
    float d2 = M2 * (LC2 * LC2 + L1 * LC2 * cosf(th2)) + I2;
    float phi2 = M2 * LC2 * G * cosf(th1 + th2 - PI_F / 2.0f);
    float phi1 = -M2 * L1 * LC2 * dth2 * dth2 * sinf(th2) - 2.0f * M2 * L1 * LC2 * dth2 * dth1 * sinf(th2)
                 + (M1 * LC1 + M2 * L1) * G * cosf(th1 - PI_F / 2.0f) + phi2;
    float ddth2 = (a + d2 / d1 * phi1 - M2 * L1 * LC2 * dth1 * dth1 * sinf(th2) - phi2)
                  / (M2 * LC2 * LC2 + I2 - d2 * d2 / d1);
    d[0] = dth1;
    d[1] = dth2;
    d[2] = -(d2 * ddth2 + phi1) / d1;
    d[3] = ddth2;
}

static float wrap(float x)
{
    while (x > PI_F) x -= 2.0f * PI_F;
    while (x < -PI_F) x += 2.0f * PI_F;
    return x;
}

static float bound(float x, float m) { return x < -m ? -m : x > m ? m : x; }

int acrobot_step(float s[4], float action)
{
    float torque = action < 0.5f ? -1.0f : action < 1.5f ? 0.0f : 1.0f;
    float k1[4], k2[4], k3[4], k4[4], y[4];
    dsdt(s, torque, k1);
    for (int i = 0; i < 4; i++) y[i] = s[i] + DT / 2.0f * k1[i];
    dsdt(y, torque, k2);
    for (int i = 0; i < 4; i++) y[i] = s[i] + DT / 2.0f * k2[i];
    dsdt(y, torque, k3);
    for (int i = 0; i < 4; i++) y[i] = s[i] + DT * k3[i];
    dsdt(y, torque, k4);
    for (int i = 0; i < 4; i++) s[i] += DT / 6.0f * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    s[0] = wrap(s[0]);
    s[1] = wrap(s[1]);
    s[2] = bound(s[2], MAX_VEL_1);
    s[3] = bound(s[3], MAX_VEL_2);
    return -cosf(s[0]) - cosf(s[1] + s[0]) > 1.0f;
}

void acrobot_obs(const float s[4], float obs[ACROBOT_OBS])
{
    obs[0] = cosf(s[0]);
    obs[1] = sinf(s[0]);
    obs[2] = cosf(s[1]);
    obs[3] = sinf(s[1]);
    obs[4] = s[2];
    obs[5] = s[3];
}
