/* Policy server over USB-Serial/JTAG. HIL step (0xA5): read an observation, run the policy, reply with the action
 * and its inference time in CPU cycles. On-chip episode (0xA6, cart-pole policies only): run the cart-pole physics
 * and the policy here at 50 Hz and stream every step. Protocol in link.py.
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <stdint.h>
#include <string.h>

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

/* One cart-pole episode from the requested initial state, one step every CARTPOLE_DT_MS ms (FreeRTOS tick aligned).
 * A frame the host does not drain within 10 ms is dropped, never waited for: the host sees the gap in step. */
static void run_episode(void)
{
    uint8_t req[1 + 16]; /* seq + 4 x f32 state */
    if (!read_exact(req, sizeof req)) return;
    float s[4];
    memcpy(s, req + 1, sizeof s);
    vTaskDelay(1); /* start on a tick edge, or step 0 -> 1 is shorter than CARTPOLE_DT_MS */
    TickType_t wake = xTaskGetTickCount();
    for (uint16_t k = 0; k < CARTPOLE_STEPS; k++) {
        uint32_t t_us = (uint32_t)esp_timer_get_time();
        uint32_t t0 = esp_cpu_get_cycle_count();
        float obs[CARTPOLE_OBS];
        cartpole_obs(s, obs);
        float action = mlp_forward(obs);
        uint8_t flags = cartpole_step(s, action) ? 1 : 0;
        uint32_t cycles = esp_cpu_get_cycle_count() - t0;
        if (k == CARTPOLE_STEPS - 1) flags |= 2;

        uint8_t f[33] = {FRAME, req[0]};
        memcpy(f + 2, &k, 2);
        memcpy(f + 4, s, 16);
        memcpy(f + 20, &action, 4);
        memcpy(f + 24, &cycles, 4);
        memcpy(f + 28, &t_us, 4);
        f[32] = flags;
        usb_serial_jtag_write_bytes(f, sizeof f, pdMS_TO_TICKS(10));
        if (flags & 1) return;
        vTaskDelayUntil(&wake, pdMS_TO_TICKS(CARTPOLE_DT_MS));
    }
}

void app_main(void)
{
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));

    for (;;) {
        uint8_t b;
        if (usb_serial_jtag_read_bytes(&b, 1, portMAX_DELAY) != 1) continue;
        if (b == EP_REQ && mlp_n_obs == CARTPOLE_OBS) run_episode();
        if (b != REQ) continue;
        uint8_t req[1 + 4 * MLP_MAX_OBS]; /* seq + mlp_n_obs x f32 */
        if (!read_exact(req, 1 + 4 * mlp_n_obs)) continue;
        float obs[MLP_MAX_OBS];
        memcpy(obs, req + 1, 4 * mlp_n_obs);

        uint32_t t0 = esp_cpu_get_cycle_count();
        float action = mlp_forward(obs);
        uint32_t cycles = esp_cpu_get_cycle_count() - t0;

        uint8_t rep[10] = {REP, req[0]};
        memcpy(rep + 2, &action, 4);
        memcpy(rep + 6, &cycles, 4);
        usb_serial_jtag_write_bytes(rep, sizeof rep, pdMS_TO_TICKS(50));
    }
}
