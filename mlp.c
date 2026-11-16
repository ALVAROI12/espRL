/* Policy forward pass; same source on the host (tests/test_mlp.py) and the ESP32. SPDX-License-Identifier: GPL-3.0-or-later */
#include "mlp.h"

#include <math.h>

#include "policy_weights.h"

_Static_assert(POLICY_OBS <= MLP_MAX_OBS && POLICY_ACT == 1, "mlp_forward returns 1 action; obs must fit MLP_MAX_OBS");

const int mlp_n_obs = POLICY_OBS;

static float h0[POLICY_HIDDEN];
static float h1[POLICY_HIDDEN];

float mlp_forward(const float *obs)
{
    for (int i = 0; i < POLICY_HIDDEN; i++) {
        float s = policy_b0[i];
        for (int j = 0; j < POLICY_OBS; j++) s += policy_w0[i][j] * obs[j];
        h0[i] = s > 0.0f ? s : 0.0f;
    }
    for (int i = 0; i < POLICY_HIDDEN; i++) {
        float s = policy_b1[i];
        for (int j = 0; j < POLICY_HIDDEN; j++) s += policy_w1[i][j] * h0[j];
        h1[i] = s > 0.0f ? s : 0.0f;
    }
    float mu = policy_b2[0];
    for (int j = 0; j < POLICY_HIDDEN; j++) mu += policy_w2[0][j] * h1[j];
    return tanhf(mu) * POLICY_ACTION_SCALE;
}
