"""Host side of the HIL serial protocol (docs/superpowers/specs/2026-11-16-esp32-rl-hil-design.md).

HIL step (physics on the Mac): request 0xA5, seq u8, POLICY_OBS x f32 obs (14 B pendulum, 22 B cart-pole). Reply: 0x5A, seq u8, POLICY_ACT x f32 actions, u32 inference cycles (10 B for 1 action, 14 B for 2).
On-chip episode (cart-pole physics on the ESP32): request 18 B: 0xA6, seq u8, 4 x f32 initial state [x, x_dot, th,
th_dot]. The chip then runs up to 500 steps at 50 Hz and sends one 33 B frame per step: 0x5B, seq u8, step u16,
4 x f32 state after the step, f32 action, u32 compute cycles (obs + policy + physics), u32 step start time (us),
u8 flags (bit 0: left the track, bit 1: last step).
"""

from __future__ import annotations

import struct
import time

import numpy as np
import serial

REQ, REP = 0xA5, 0x5A
EP_REQ, FRAME = 0xA6, 0x5B
FRAME_FMT = "<BBH4ffIIB"
FRAME_SIZE = struct.calcsize(FRAME_FMT)  # 33


class Link:
    def __init__(self, port: str, timeout: float = 0.1, serial_obj=None, n_act: int = 1):
        if serial_obj is None:
            # Baud is ignored by USB-Serial/JTAG. write_timeout: a wedged board that stops draining must not hang us.
            serial_obj = serial.Serial(port, 115200, timeout=0.005, write_timeout=timeout)
        self.ser = serial_obj
        self.timeout = timeout
        self.seq = 0
        self.n_act = n_act
        self.buf = bytearray()
        self.ser.reset_input_buffer()

    def act(self, obs: np.ndarray) -> tuple[tuple[float, ...], int] | None:
        """Send one observation; return (actions, inference cycles), or None if no matching reply in time."""
        seq = self.seq
        self.seq = (self.seq + 1) & 0xFF
        o = np.asarray(obs, dtype=np.float32)
        try:
            self.ser.write(struct.pack(f"<BB{o.size}f", REQ, seq, *o))
        except serial.SerialTimeoutException:
            return None
        r = self._recv(REP, f"<BB{self.n_act}fI", self.timeout, seq)
        return None if r is None else (tuple(r[:-1]), r[-1])

    def start_episode(self, state: np.ndarray) -> int | None:
        """Ask the chip to run one cart-pole episode from state; returns the seq its frames carry (None: write timeout)."""
        seq = self.seq
        self.seq = (self.seq + 1) & 0xFF
        try:
            self.ser.write(struct.pack("<BB4f", EP_REQ, seq, *np.asarray(state, dtype=np.float32)))
        except serial.SerialTimeoutException:
            return None
        return seq

    def frame(self, seq: int, timeout: float | None = None) -> tuple | None:
        """Next frame of episode seq: (step, state 4-tuple, action, cycles, t_us, flags), or None if none in time."""
        r = self._recv(FRAME, FRAME_FMT, self.timeout if timeout is None else timeout, seq)
        return None if r is None else (r[0], tuple(r[1:5]), *r[5:])

    def _recv(self, marker: int, fmt: str, timeout: float, seq: int) -> tuple | None:
        """Fields after (marker, seq) of the next message with this marker and seq; skips junk and stale seqs."""
        size = struct.calcsize(fmt)
        deadline = time.monotonic() + timeout
        while True:
            i = self.buf.find(marker)
            if i < 0:
                self.buf.clear()
            elif len(self.buf) - i >= size:
                fields = struct.unpack_from(fmt, self.buf, i)
                if fields[1] == seq:
                    del self.buf[:i + size]
                    return fields[2:]
                del self.buf[:i + 1]  # stale message or a marker byte inside junk: rescan from the next byte
                continue
            if time.monotonic() > deadline:
                return None
            # Ask for exactly the missing bytes: pyserial's read(n) waits out its timeout until n bytes arrive.
            self.buf += self.ser.read(size if i < 0 else i + size - len(self.buf))

    def close(self) -> None:
        self.ser.close()
