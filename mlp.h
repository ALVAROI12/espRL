/* Policy forward pass (SAC actor exported by export.py). SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

/* obs = [cos th, sin th, th_dot]; returns the deterministic action tanh(mu) * POLICY_ACTION_SCALE. */
float mlp_forward(const float obs[3]);
