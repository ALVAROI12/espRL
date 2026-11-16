"""Host side of the HIL serial protocol (link.py) against a fake port."""

import struct
import time

import numpy as np
import serial

from link import Link


class FakeSerial:
    def __init__(self, read_timeout=0.0):
        self.rx = bytearray()   # bytes the "board" will send
        self.tx = bytearray()   # bytes the host wrote
        self.read_timeout = read_timeout  # like pyserial: read(n) waits this long when fewer than n bytes are ready

    def write(self, data):
        self.tx += data
        return len(data)

    def read(self, n):
        if len(self.rx) < n:
            time.sleep(self.read_timeout)
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


def test_reply_does_not_wait_for_read_timeout():
    fake = FakeSerial(read_timeout=0.005)
    link = Link("fake", timeout=0.1, serial_obj=fake)
    fake.rx += reply(0, 0.5, 1)
    t0 = time.perf_counter()
    assert link.act(np.zeros(3)) == (0.5, 1)
    assert time.perf_counter() - t0 < 0.003  # a full 10-byte reply is ready: no 5 ms read timeout in the RTT


def test_write_timeout_returns_none():
    link, fake = make()

    def stuck(_):
        raise serial.SerialTimeoutException("Write timeout")
    fake.write = stuck
    assert link.act(np.zeros(3)) is None


def test_request_bytes_cartpole():
    link, fake = make()
    fake.rx += reply(0, 0.0, 0)
    obs = np.array([0.1, -0.2, 1.0, 0.0, 0.5])
    link.act(obs)
    assert bytes(fake.tx) == struct.pack("<BB5f", 0xA5, 0, *obs) and len(fake.tx) == 22


def frame(seq, step, state, action=0.0, cycles=0, t_us=0, flags=0):
    return struct.pack("<BBH4ffIIB", 0x5B, seq, step, *state, action, cycles, t_us, flags)


def test_episode_request_and_frames():
    link, fake = make()
    seq = link.start_episode(np.array([0.1, 0.0, 3.1, 0.0]))
    assert bytes(fake.tx) == struct.pack("<BB4f", 0xA6, 0, 0.1, 0.0, 3.1, 0.0) and len(fake.tx) == 18
    fake.rx += b"junk" + frame(9, 0, (9, 9, 9, 9)) + frame(seq, 0, (1, 2, 3, 4), 0.5, 7, 100, 0) + frame(seq, 1, (0, 0, 0, 0), flags=2)
    step, state, action, cycles, t_us, flags = link.frame(seq)
    assert (step, state, action, cycles, t_us, flags) == (0, (1.0, 2.0, 3.0, 4.0), 0.5, 7, 100, 0)
    assert link.frame(seq)[0] == 1
    assert link.frame(seq) is None  # nothing more: timeout
