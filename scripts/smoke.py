#!/usr/bin/env python3
"""Read N frames from the audio-tap socket and print a summary."""
import socket
import struct
import sys

SOCK = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rctx.sock"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def read_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("server closed")
        buf += chunk
    return buf


def main():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(SOCK)

    counts = {0: 0, 1: 0}
    first_ts = {0: None, 1: None}
    last_ts = {0: None, 1: None}
    payload_sizes = []

    for i in range(N):
        hdr = read_exact(s, 9)
        tag, ts_ms, payload_len = struct.unpack(">BII", hdr)
        _ = read_exact(s, payload_len)
        counts[tag] = counts.get(tag, 0) + 1
        if first_ts.get(tag) is None:
            first_ts[tag] = ts_ms
        last_ts[tag] = ts_ms
        payload_sizes.append(payload_len)

    print(f"frames read: {N}")
    print(f"  them (system audio): {counts.get(0, 0)}")
    print(f"  me   (mic):          {counts.get(1, 0)}")
    if payload_sizes:
        print(f"  payload bytes: min={min(payload_sizes)} max={max(payload_sizes)} avg={sum(payload_sizes) // len(payload_sizes)}")
    for tag, name in [(0, "them"), (1, "me")]:
        if first_ts.get(tag) is not None:
            span = last_ts[tag] - first_ts[tag]
            print(f"  {name} timestamps: {first_ts[tag]}ms → {last_ts[tag]}ms (span={span}ms)")


if __name__ == "__main__":
    main()
