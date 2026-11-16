"""Host side of the HIL serial protocol (rl/link.py) against a fake port."""

import struct

import numpy as np

from link import Link


class FakeSerial:
    def __init__(self):
        self.rx = bytearray()   # bytes the "board" will send
        self.tx = bytearray()   # bytes the host wrote

    def write(self, data):
        self.tx += data
        return len(data)

    def read(self, n):
        out, self.rx = bytes(self.rx[:n]), self.rx[n:]
        return out

    def reset_input_buffer(self):
        pass

    def close(self):
        pass


def reply(seq, action, cycles):
    return struct.pack("<BBfI", 0x5A, seq, action, cycles)


def make():
    fake = FakeSerial()
    return Link("fake", timeout=0.05, serial_obj=fake), fake


def test_reply_after_junk():
    link, fake = make()
    fake.rx += b"ESP-ROM:esp32s3\r\n" + reply(0, 0.5, 1234)
    assert link.act(np.zeros(3)) == (0.5, 1234)


def test_stale_seq_skipped():
    link, fake = make()
    fake.rx += reply(7, -1.0, 1) + reply(0, 0.25, 2)
    assert link.act(np.zeros(3)) == (0.25, 2)


def test_timeout_returns_none():
    link, _ = make()
    assert link.act(np.zeros(3)) is None


def test_seq_wraps():
    link, fake = make()
    for i in range(300):
        fake.rx += reply(i % 256, 1.0, i)
        assert link.act(np.zeros(3)) == (1.0, i)


def test_request_bytes():
    link, fake = make()
    fake.rx += reply(0, 0.0, 0)
    link.act(np.array([1.0, 0.0, 0.0]))
    assert bytes(fake.tx) == struct.pack("<BB3f", 0xA5, 0, 1.0, 0.0, 0.0)
