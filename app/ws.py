"""Tiny RFC6455 WebSocket server (stdlib only, board has no `websockets`)."""
from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import threading

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WsClosed(Exception):
    pass


class WsConnection:
    def __init__(self, handler):
        self.handler = handler
        self.rfile = handler.rfile
        self.wfile = handler.wfile
        self.sock = handler.connection
        self._send_lock = threading.Lock()
        self.closed = False
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def handshake(self):
        key = self.handler.headers.get("Sec-WebSocket-Key", "")
        if not key:
            raise WsClosed("missing Sec-WebSocket-Key")
        accept = base64.b64encode(
            hashlib.sha1((key + GUID).encode()).digest()).decode()
        self.wfile.write(
            ("HTTP/1.1 101 Switching Protocols\r\n"
             "Upgrade: websocket\r\n"
             "Connection: Upgrade\r\n"
             f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
        self.wfile.flush()

    # ------------------------------------------------------------------ recv
    def _read_exact(self, n: int) -> bytes:
        data = self.rfile.read(n)
        if data is None or len(data) < n:
            raise WsClosed("short read")
        return data

    def recv(self):
        """Return ('text', str) | ('binary', bytes) | ('close', None)."""
        frag_op = None
        frag = bytearray()
        while True:
            try:
                b1, b2 = self._read_exact(2)
            except (WsClosed, OSError):
                raise WsClosed("connection closed")
            fin = b1 & 0x80
            opcode = b1 & 0x0F
            masked = b2 & 0x80
            length = b2 & 0x7F
            if length == 126:
                (length,) = struct.unpack("!H", self._read_exact(2))
            elif length == 127:
                (length,) = struct.unpack("!Q", self._read_exact(8))
            if length > 8 * 1024 * 1024:
                raise WsClosed("frame too large")
            mask = self._read_exact(4) if masked else b"\x00" * 4
            payload = bytearray(self._read_exact(length)) if length else bytearray()
            if masked:
                for i in range(length):
                    payload[i] ^= mask[i & 3]

            if opcode == 0x8:
                return "close", None
            if opcode == 0x9:
                self._send_frame(0xA, bytes(payload))
                continue
            if opcode == 0xA:
                continue
            if opcode in (0x1, 0x2):
                frag_op = opcode
                frag = payload
            elif opcode == 0x0:
                frag.extend(payload)
            else:
                continue
            if not fin:
                continue
            if frag_op == 0x1:
                return "text", bytes(frag).decode("utf-8", "replace")
            if frag_op == 0x2:
                return "binary", bytes(frag)

    # ------------------------------------------------------------------ send
    def _send_frame(self, opcode: int, payload: bytes):
        header = bytearray([0x80 | opcode])
        n = len(payload)
        if n < 126:
            header.append(n)
        elif n < 65536:
            header.append(126)
            header += struct.pack("!H", n)
        else:
            header.append(127)
            header += struct.pack("!Q", n)
        with self._send_lock:
            if self.closed:
                return
            try:
                self.wfile.write(bytes(header) + payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.closed = True
                raise WsClosed("write failed")

    def send_text(self, text: str):
        self._send_frame(0x1, text.encode("utf-8"))

    def send_json(self, obj):
        self.send_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))

    def send_bytes(self, data: bytes):
        self._send_frame(0x2, data)

    def close(self):
        if self.closed:
            return
        try:
            self._send_frame(0x8, b"")
        except Exception:
            pass
        self.closed = True
