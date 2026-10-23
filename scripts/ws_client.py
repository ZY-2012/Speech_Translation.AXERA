#!/usr/bin/env python3
"""Minimal WS client used to smoke-test the demo without a browser."""
import base64
import hashlib
import json
import os
import socket
import struct
import sys
import time

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8091


def send_frame(sock, opcode, payload):
    header = bytearray([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header.append(0x80 | n)
    elif n < 65536:
        header.append(0x80 | 126)
        header += struct.pack("!H", n)
    else:
        header.append(0x80 | 127)
        header += struct.pack("!Q", n)
    mask = os.urandom(4)
    header += mask
    body = bytearray(payload)
    for i in range(n):
        body[i] ^= mask[i & 3]
    sock.sendall(bytes(header) + bytes(body))


def read_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def recv_frame(sock):
    b1, b2 = read_exact(sock, 2)
    opcode = b1 & 0x0F
    length = b2 & 0x7F
    if length == 126:
        (length,) = struct.unpack("!H", read_exact(sock, 2))
    elif length == 127:
        (length,) = struct.unpack("!Q", read_exact(sock, 8))
    payload = read_exact(sock, length) if length else b""
    return opcode, payload


def main():
    key = base64.b64encode(os.urandom(16)).decode()
    s = socket.create_connection((HOST, PORT), timeout=5)
    s.sendall((
        f"GET /ws HTTP/1.1\r\nHost: {HOST}:{PORT}\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    ).encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += s.recv(1)
    print(resp.decode().splitlines()[0])
    s.settimeout(0.2)

    def drain(duration):
        end = time.time() + duration
        while time.time() < end:
            try:
                op, data = recv_frame(s)
            except socket.timeout:
                continue
            if op == 0x1:
                print("event:", data.decode())
            elif op == 0x2:
                print("audio:", len(data), "bytes, header", data[0])
            elif op == 0x8:
                print("closed")
                return False
        return True

    drain(0.5)
    send_frame(s, 0x1, json.dumps({"type": "start"}).encode())
    drain(0.5)
    # one second of a loud 440 Hz tone -> mock energy VAD fires
    import math
    rate = 16000
    chunk = 320
    for i in range(rate // chunk):
        pcm = bytearray()
        for j in range(chunk):
            v = int(10000 * math.sin(2 * math.pi * 440 * (i * chunk + j) / rate))
            pcm += struct.pack("<h", v)
        send_frame(s, 0x2, bytes(pcm))
        time.sleep(0.02)
    # silence to close the utterance
    for i in range(60):
        send_frame(s, 0x2, b"\x00\x00" * chunk)
        time.sleep(0.02)
    drain(4.0)
    send_frame(s, 0x1, json.dumps({"type": "stop"}).encode())
    drain(1.0)
    s.close()


if __name__ == "__main__":
    main()
