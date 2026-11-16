"""Host side of the HIL serial protocol (docs/superpowers/specs/2026-11-16-esp32-rl-hil-design.md).

Request 14 B: 0xA5, seq u8, 3 x f32 obs. Reply 10 B: 0x5A, seq u8, f32 action, u32 inference cycles.
"""

from __future__ import annotations

import struct
import time

import numpy as np

REQ, REP = 0xA5, 0x5A
REPLY_SIZE = 10


class Link:
    def __init__(self, port: str, timeout: float = 0.1, serial_obj=None):
        if serial_obj is None:
            import serial
            serial_obj = serial.Serial(port, 115200, timeout=0.005)  # baud is ignored by USB-Serial/JTAG
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
        self.ser.write(struct.pack("<BB3f", REQ, seq, *o))
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
            self.buf += self.ser.read(64)

    def close(self) -> None:
        self.ser.close()
