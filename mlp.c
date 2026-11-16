/* Policy forward pass; same source on the host (tests/test_mlp.py) and the ESP32. SPDX-License-Identifier: GPL-3.0-or-later */
#include "mlp.h"

#include <math.h>

#include "policy_weights.h"

_Static_assert(POLICY_OBS <= MLP_MAX_OBS && POLICY_ACT <= MLP_MAX_ACT, "policy must fit MLP_MAX_OBS / MLP_MAX_ACT");

const int mlp_n_obs = POLICY_OBS;
const int mlp_n_act = POLICY_ACT;
const char mlp_env[] = POLICY_ENV;

static float h0[POLICY_HIDDEN];
static float h1[POLICY_HIDDEN];
static float out[POLICY_OUT];

void mlp_forward(const float *obs, float *act)
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
    for (int i = 0; i < POLICY_OUT; i++) {
        float s = policy_b2[i];
        for (int j = 0; j < POLICY_HIDDEN; j++) s += policy_w2[i][j] * h1[j];
        out[i] = s;
    }
#if POLICY_DISCRETE
    int best = 0; /* argmax Q; strict > keeps the first maximum, like torch.argmax */
    for (int i = 1; i < POLICY_OUT; i++)
        if (out[i] > out[best]) best = i;
    act[0] = (float)best;
#else
    for (int i = 0; i < POLICY_ACT; i++) act[i] = tanhf(out[i]) * POLICY_ACTION_SCALE;
#endif
}
