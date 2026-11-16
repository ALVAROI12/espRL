/* Policy forward pass (SAC actor exported by export.py). SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#define MLP_MAX_OBS 8 /* request / reply buffer sizes in firmware/main/main.c; mlp.c asserts the policy fits */
#define MLP_MAX_ACT 4

/* Observations / actions of the exported policy (POLICY_OBS / POLICY_ACT), for code without policy_weights.h. */
extern const int mlp_n_obs;
extern const int mlp_n_act;

/* obs = mlp_n_obs floats in the env's order; writes the mlp_n_act deterministic actions tanh(mu) * POLICY_ACTION_SCALE. */
void mlp_forward(const float *obs, float *act);
