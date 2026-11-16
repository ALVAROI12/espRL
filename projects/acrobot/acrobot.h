/* Acrobot physics on the chip: float32 port of Gymnasium's Acrobot-v1 ("book" dynamics, RK4, no torque noise).
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#define ACROBOT_OBS 6
#define ACROBOT_STEPS 500 /* Acrobot-v1 TimeLimit */
#define ACROBOT_DT_MS 200

/* s = [th1, th2, dth1, dth2]; action = index 0, 1, 2 for torque -1, 0, +1 (rounded, clipped).
 * Advances s by one 200 ms step; returns 1 when the tip is above the goal line (the episode ends). */
int acrobot_step(float s[4], float action);

/* obs = [cos th1, sin th1, cos th2, sin th2, dth1, dth2], the policy's input. */
void acrobot_obs(const float s[4], float obs[ACROBOT_OBS]);
