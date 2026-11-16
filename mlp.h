/* Policy forward pass (SAC actor exported by export.py). SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#define MLP_MAX_OBS 8 /* request buffer size in firmware/main/main.c; mlp.c asserts POLICY_OBS fits */

/* Observations the exported policy takes (POLICY_OBS), for code that does not include policy_weights.h. */
extern const int mlp_n_obs;

/* obs = mlp_n_obs floats in the env's order; returns the deterministic action tanh(mu) * POLICY_ACTION_SCALE. */
float mlp_forward(const float *obs);
