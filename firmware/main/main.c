/* Policy server over USB-Serial/JTAG. HIL step (0xA5): read an observation, run the policy, reply with the action(s)
 * and the inference time in CPU cycles. On-chip episode (0xA6, envs in SIMS below): run the env's physics and the
 * policy here in real time and stream every step. Protocol in link.py.
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <stdint.h>
#include <string.h>

#include "acrobot.h"
#include "cartpole.h"
#include "driver/usb_serial_jtag.h"
#include "esp_cpu.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "mlp.h"

#define REQ 0xA5
#define REP 0x5A
#define EP_REQ 0xA6
#define FRAME 0x5B

static int read_exact(uint8_t *buf, int len)
{
    int got = 0;
    while (got < len) {
        int n = usb_serial_jtag_read_bytes(buf + got, len - got, pdMS_TO_TICKS(50));
        if (n <= 0) return 0; /* stalled mid-request: drop it and resync on the next 0xA5 */
        got += n;
    }
    return 1;
}

/* Envs whose physics runs on the chip; the flashed policy's POLICY_ENV picks one by prefix. State is 4 floats. */
struct sim {
    const char *env_prefix;
    int n_obs, steps, dt_ms;
    int (*step)(float s[4], float action); /* returns 1 when the episode terminates */
    void (*obs)(const float s[4], float *obs);
};
static const struct sim SIMS[] = {
    {"cartpole_", CARTPOLE_OBS, CARTPOLE_STEPS, CARTPOLE_DT_MS, cartpole_step, cartpole_obs},
    {"acrobot", ACROBOT_OBS, ACROBOT_STEPS, ACROBOT_DT_MS, acrobot_step, acrobot_obs},
};

static const struct sim *find_sim(void)
{
    for (size_t i = 0; i < sizeof SIMS / sizeof SIMS[0]; i++)
        if (strncmp(mlp_env, SIMS[i].env_prefix, strlen(SIMS[i].env_prefix)) == 0 && mlp_n_obs == SIMS[i].n_obs &&
            mlp_n_act == 1)
            return &SIMS[i];
    return NULL;
}

/* One episode from the requested initial state, one step every dt_ms (FreeRTOS tick aligned).
 * A frame the host does not drain within 10 ms is dropped, never waited for: the host sees the gap in step. */
static void run_episode(const struct sim *sim)
{
    uint8_t req[1 + 16]; /* seq + 4 x f32 state */
    if (!read_exact(req, sizeof req)) return;
    float s[4];
    memcpy(s, req + 1, sizeof s);
    vTaskDelay(1); /* start on a tick edge, or step 0 -> 1 is shorter than dt_ms */
    TickType_t wake = xTaskGetTickCount();
    for (uint16_t k = 0; k < sim->steps; k++) {
        uint32_t t_us = (uint32_t)esp_timer_get_time();
        uint32_t t0 = esp_cpu_get_cycle_count();
        float obs[MLP_MAX_OBS];
        sim->obs(s, obs);
        float act[MLP_MAX_ACT];
        mlp_forward(obs, act);
        float action = act[0]; /* force / 10 (cart-pole) or action index (acrobot) */
        uint8_t flags = sim->step(s, action) ? 1 : 0;
        uint32_t cycles = esp_cpu_get_cycle_count() - t0;
        if (k == sim->steps - 1) flags |= 2;

        uint8_t f[33] = {FRAME, req[0]};
        memcpy(f + 2, &k, 2);
        memcpy(f + 4, s, 16);
        memcpy(f + 20, &action, 4);
        memcpy(f + 24, &cycles, 4);
        memcpy(f + 28, &t_us, 4);
        f[32] = flags;
        usb_serial_jtag_write_bytes(f, sizeof f, pdMS_TO_TICKS(10));
        if (flags & 1) return;
        vTaskDelayUntil(&wake, pdMS_TO_TICKS(sim->dt_ms));
    }
}

void app_main(void)
{
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));

    for (;;) {
        uint8_t b;
        if (usb_serial_jtag_read_bytes(&b, 1, portMAX_DELAY) != 1) continue;
        if (b == EP_REQ && find_sim()) run_episode(find_sim());
        if (b != REQ) continue;
        uint8_t req[1 + 4 * MLP_MAX_OBS]; /* seq + mlp_n_obs x f32 */
        if (!read_exact(req, 1 + 4 * mlp_n_obs)) continue;
        float obs[MLP_MAX_OBS];
        memcpy(obs, req + 1, 4 * mlp_n_obs);

        uint32_t t0 = esp_cpu_get_cycle_count();
        float act[MLP_MAX_ACT];
        mlp_forward(obs, act);
        uint32_t cycles = esp_cpu_get_cycle_count() - t0;

        uint8_t rep[2 + 4 * MLP_MAX_ACT + 4] = {REP, req[0]}; /* seq, mlp_n_act x f32, cycles */
        memcpy(rep + 2, act, 4 * mlp_n_act);
        memcpy(rep + 2 + 4 * mlp_n_act, &cycles, 4);
        usb_serial_jtag_write_bytes(rep, 2 + 4 * mlp_n_act + 4, pdMS_TO_TICKS(50));
    }
}
