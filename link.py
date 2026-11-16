"""Host side of the HIL serial protocol (docs/superpowers/specs/2026-11-16-esp32-rl-hil-design.md).

Request: 0xA5, seq u8, POLICY_OBS x f32 obs (14 B pendulum, 22 B cart-pole). Reply 10 B: 0x5A, seq u8, f32 action, u32 inference cycles.
"""

from __future__ import annotations

import struct
import time

import numpy as np
import serial

REQ, REP = 0xA5, 0x5A
REPLY_SIZE = 10


class Link:
    def __init__(self, port: str, timeout: float = 0.1, serial_obj=None):
        if serial_obj is None:
            # Baud is ignored by USB-Serial/JTAG. write_timeout: a wedged board that stops draining must not hang us.
            serial_obj = serial.Serial(port, 115200, timeout=0.005, write_timeout=timeout)
        self.ser = serial_obj
        self.timeout = timeout
        self.seq = 0
        self.buf = bytearray()
        self.ser.reset_input_buffer()

    def act(self, obs: np.ndarray) -> tuple[float, int] | None:
        """Send one observation; return (action, inference cycles), or None if no matching reply in time."""
        seq = self.seq
        self.seq = (self.seq + 1) & 0xFF
        o = np.asarray(obs, dtype=np.float32)
        try:
            self.ser.write(struct.pack(f"<BB{o.size}f", REQ, seq, *o))
        except serial.SerialTimeoutException:
            return None
        deadline = time.monotonic() + self.timeout
        while True:
            i = self.buf.find(REP)
            if i < 0:
                self.buf.clear()
            elif len(self.buf) - i >= REPLY_SIZE:
                _, rseq, action, cycles = struct.unpack_from("<BBfI", self.buf, i)
                if rseq == seq:
                    del self.buf[:i + REPLY_SIZE]
                    return action, cycles
                del self.buf[:i + 1]  # stale reply or a 0x5A inside junk: rescan from the next byte
                continue
            if time.monotonic() > deadline:
                return None
            # Ask for exactly the missing bytes: pyserial's read(n) waits out its timeout until n bytes arrive.
            self.buf += self.ser.read(REPLY_SIZE if i < 0 else i + REPLY_SIZE - len(self.buf))

    def close(self) -> None:
        self.ser.close()
