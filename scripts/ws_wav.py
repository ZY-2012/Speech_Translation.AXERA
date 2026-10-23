#!/usr/bin/env python3
"""Stream a 16 kHz WAV as microphone PCM to the running web demo."""
import json
import os
import sys
import time
import wave

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from ws_client import send_frame, recv_frame, socket  # noqa: E402

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8095
WAV = sys.argv[3] if len(sys.argv) > 3 else str(__import__("pathlib").Path(__file__).resolve().parent.parent / "samples/vad_example.wav")
SECONDS = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0


def main():
    import base64
    import os
    key = base64.b64encode(os.urandom(16)).decode()
    s = socket.create_connection((HOST, PORT), timeout=5)
    if os.environ.get("WSS_TLS") == "1" or PORT == 8443:
        import ssl
        s = ssl._create_unverified_context().wrap_socket(s, server_hostname=HOST)
    s.sendall((f"GET /ws HTTP/1.1\r\nHost: {HOST}:{PORT}\r\n"
               "Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += s.recv(1)
    print("connected", resp.split(b"\r\n")[0].decode())
    s.settimeout(0.005)
    send_frame(s, 0x1, json.dumps({"type": "start"}).encode())

    with wave.open(WAV, "rb") as wf:
        rate = wf.getframerate()
        ch = wf.getnchannels()
        raw = wf.readframes(min(wf.getnframes(), int(rate * SECONDS)))
    if ch != 1 or rate != 16000:
        print("WAV must be 16k mono", rate, ch)
        return
    step = 640  # 20 ms
    t0 = time.time()
    audio_frames = 0
    events = []
    for i in range(0, len(raw), step):
        chunk = raw[i:i + step]
        send_frame(s, 0x2, chunk)
        # drain events without blocking
        end = time.time() + 0.02
        while time.time() < end:
            try:
                op, data = recv_frame(s)
            except socket.timeout:
                break
            if op == 0x1:
                msg = json.loads(data)
                typ = msg.get("type")
                if typ in ("asr_partial", "asr_final", "trans_final",
                           "trans_token", "tts_first_frame", "tts_end",
                           "speech_start", "speech_end", "error"):
                    if typ == "trans_token":
                        continue
                    events.append(msg)
                    print(f"[{time.time()-t0:6.2f}] {typ}: "
                          f"{str({k:v for k,v in msg.items() if k not in ('type','t')})[:220]}",
                          flush=True)
            elif op == 0x2:
                audio_frames += 1
        # real-time pacing
        target = t0 + (i + step) / (16000 * 2)
        if target > time.time():
            time.sleep(target - time.time())
    # trailing silence to close the last utterance
    for _ in range(70):
        send_frame(s, 0x2, b"\x00\x00" * 320)
        time.sleep(0.02)
    end = time.time() + 25
    while time.time() < end:
        try:
            op, data = recv_frame(s)
        except socket.timeout:
            continue
        if op == 0x1:
            msg = json.loads(data)
            if msg.get("type") in ("asr_final", "trans_final", "tts_end",
                                   "tts_first_frame", "error"):
                print(f"[{time.time()-t0:6.2f}] {msg.get('type')}: "
                      f"{str({k:v for k,v in msg.items() if k not in ('type','t')})[:240]}",
                      flush=True)
        elif op == 0x2:
            audio_frames += 1
    print("audio frames received:", audio_frames)
    send_frame(s, 0x1, json.dumps({"type": "stop"}).encode())
    s.close()


if __name__ == "__main__":
    main()
