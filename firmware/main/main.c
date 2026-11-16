/* HIL policy server: read an observation over USB-Serial/JTAG, run the policy, reply with the action and its
 * inference time in CPU cycles. Protocol in docs/superpowers/specs/2026-11-16-esp32-rl-hil-design.md.
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <stdint.h>
#include <string.h>

#include "driver/usb_serial_jtag.h"
#include "esp_cpu.h"
#include "freertos/FreeRTOS.h"
#include "mlp.h"

#define REQ 0xA5
#define REP 0x5A

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

void app_main(void)
{
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));

    for (;;) {
        uint8_t b;
        if (usb_serial_jtag_read_bytes(&b, 1, portMAX_DELAY) != 1 || b != REQ) continue;
        uint8_t req[13]; /* seq + 3 x f32 */
        if (!read_exact(req, sizeof req)) continue;
        float obs[3];
        memcpy(obs, req + 1, sizeof obs);

        uint32_t t0 = esp_cpu_get_cycle_count();
        float action = mlp_forward(obs);
        uint32_t cycles = esp_cpu_get_cycle_count() - t0;

        uint8_t rep[10] = {REP, req[0]};
        memcpy(rep + 2, &action, 4);
        memcpy(rep + 6, &cycles, 4);
        usb_serial_jtag_write_bytes(rep, sizeof rep, pdMS_TO_TICKS(50));
    }
}
