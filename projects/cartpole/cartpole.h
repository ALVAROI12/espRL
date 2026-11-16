/* Cart-pole physics on the chip: float32 port of cartpole.py. SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#define CARTPOLE_OBS 5
#define CARTPOLE_STEPS 500 /* episode length, as registered in cartpole.py */
#define CARTPOLE_DT_MS 20

/* s = [x, x_dot, th, th_dot], th = 0 upright; action in [-1, 1] (clipped) is force / 10 N.
 * Advances s by one 20 ms step; returns 1 if the cart left the track (the episode ends). */
int cartpole_step(float s[4], float action);

/* obs = [x, x_dot, cos th, sin th, th_dot], the policy's input. */
void cartpole_obs(const float s[4], float obs[CARTPOLE_OBS]);
