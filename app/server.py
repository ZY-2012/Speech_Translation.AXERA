#!/usr/bin/env python3
"""Streaming speech-translation web demo (browser microphone -> subtitles + TTS).

Run on the board:
    python3 app/server.py --config config/board.yaml
then open http://<board-ip>:8096/.
"""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import re
import signal
import sys
import threading
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config as config_mod  # noqa: E402
from app.pipeline import Components, Session  # noqa: E402
from app.ws import WsClosed, WsConnection  # noqa: E402

STATE = {
    "cfg": None,
    "components": None,
    "mock": False,
    "session": None,
    "tls_port": 0,
    "lock": threading.Lock(),
}


def _status():
    c = STATE["components"]
    cfg = STATE.get("cfg") or {}
    return {
        "mock": STATE["mock"],
        "direction": cfg.get("llm", {}).get("direction", "auto"),
        "barge_in": bool(cfg.get("runtime", {}).get("barge_in", False)),
        "asr_stream": bool(c.asr_stream and c.asr_stream.ready()) if c else False,
        "asr_offline": bool(c.asr and c.asr.ready()) if c else False,
        "asr_mode": getattr(c, "asr_mode", "hybrid") if c else "hybrid",
        "tts_enabled": bool(getattr(c, "tts_enabled", True)) if c else True,
        "tts_continuous": bool(getattr(c, "tts_continuous", True)) if c else True,
        "tts_voice": getattr(c, "tts_voice", "auto") if c else "auto",
        "llm": bool(c.llm and c.llm.ready()) if c else False,
        "mt_backend": getattr(c, "mt_backend", None) if c else None,
        "qwen": bool(c.llm_qwen and c.llm_qwen.ready()) if c else False,
        "opus": bool(c.llm_opus and c.llm_opus.ready()) if c else False,
        "tts": bool(c and ((c.tts and c.tts.ready()) or
                           (getattr(c, "tts_zh", None) and c.tts_zh.ready()) or
                           (getattr(c, "tts_custom", None) and c.tts_custom.ready()))),
        "tts_zh": bool(getattr(c, "tts_zh", None) and c.tts_zh.ready()) if c else False,
        "tts_custom": bool(getattr(c, "tts_custom", None) and c.tts_custom.ready()) if c else False,
        "tts_custom_loading": bool(getattr(c, "tts_custom_loading", False)) if c else False,
        "tts_custom_error": getattr(c, "tts_custom_error", "") if c else "",
        "tts_custom_name": getattr(c, "tts_custom_name", "") if c else "",
        "session": bool(STATE["session"]),
        "tls_port": int(STATE.get("tls_port", 0)),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SpeechTranslationZipVoice/1.0"

    def log_message(self, fmt, *args):
        print("[http] " + fmt % args, flush=True)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/ws":
            self._ws()
            return
        if path in ("/", "/index.html"):
            self._serve(ROOT / "static/index.html")
            return
        if path == "/health":
            self._json(_status())
            return
        if path == "/api/samples":
            self._json(self._samples())
            return
        if path.startswith("/samples/"):
            rel = unquote(path[len("/samples/"):])
            if not rel.lower().endswith(".wav") or "/" in rel or ".." in rel:
                self.send_error(404)
                return
            target = (ROOT / "samples" / rel).resolve()
            if ROOT / "samples" not in target.parents:
                self.send_error(403)
                return
            self._serve(target)
            return
        if path.startswith("/static/"):
            rel = unquote(path[len("/static/"):])
            target = (ROOT / "static" / rel).resolve()
            if ROOT / "static" not in target.parents:
                self.send_error(403)
                return
            self._serve(target)
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/voice":
            self.send_error(404)
            return
        c = STATE["components"]
        if c is None:
            self.close_connection = True
            self._json({"ok": False, "error": "components not ready"}, 503)
            return
        with STATE["lock"]:
            if STATE["session"] is not None:
                self.close_connection = True
                self._json({"ok": False, "error": "stop the active session first"}, 409)
                return
        if getattr(c, "tts_custom_loading", False):
            self.close_connection = True
            self._json({"ok": False, "error": "voice is already loading"}, 409)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 20 * 1024 * 1024:
            self.close_connection = True
            self._json({"ok": False, "error": "WAV must be 1-20 MB"}, 400)
            return
        body = self.rfile.read(length)
        qs = parse_qs(parsed.query)
        text = (qs.get("text", [""])[0] or "").strip()
        text = " ".join(text.split())
        name = (qs.get("name", ["reference.wav"])[0] or "reference.wav").strip()
        name = re.sub(r"[^\w.\-]+", "_", name)[:80] or "reference.wav"
        if not text or len(text) > 120:
            self._json({"ok": False, "error": "reference text is required (1-120 chars)"}, 400)
            return
        try:
            with wave.open(io.BytesIO(body), "rb") as wf:
                channels = wf.getnchannels()
                width = wf.getsampwidth()
                rate = wf.getframerate()
                frames = wf.getnframes()
            duration = frames / float(rate or 1)
        except Exception:
            self._json({"ok": False, "error": "invalid WAV file"}, 400)
            return
        if channels not in (1, 2):
            self._json({"ok": False, "error": "WAV must be mono or stereo"}, 400)
            return
        if width not in (2, 4):
            self._json({"ok": False, "error": "WAV must be 16-bit or 32-bit"}, 400)
            return
        if not (3.0 <= duration <= 5.0):
            self._json({"ok": False, "error": "reference WAV must be 3-5 seconds"}, 400)
            return
        out_dir = ROOT / ".work" / "voices"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "custom.wav"
        try:
            out.write_bytes(body)
        except Exception:
            traceback.print_exc()
            self._json({"ok": False, "error": "cannot save reference WAV"}, 500)
            return
        result = c.load_custom_voice(out, text, name)
        if not result.get("ok"):
            self._json(result, 409)
            return
        result.update({"duration": round(duration, 2), "rate": rate,
                       "channels": channels})
        self._json(result)

    @staticmethod
    def _samples():
        import wave
        out = []
        for p in sorted((ROOT / "samples").glob("*.wav")):
            try:
                with wave.open(str(p), "rb") as wf:
                    duration = wf.getnframes() / float(wf.getframerate())
                    rate = wf.getframerate()
                    channels = wf.getnchannels()
            except Exception:
                duration, rate, channels = 0.0, 0, 0
            out.append({"name": p.name, "url": "/samples/" + p.name,
                        "duration": round(duration, 1), "rate": rate,
                        "channels": channels})
        return out

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve(self, path: Path):
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".html":
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------ websocket
    def _ws(self):
        cfg = STATE["cfg"]
        with STATE["lock"]:
            if STATE["session"] is not None:
                self.send_error(409, "another session is active")
                return
        ws = WsConnection(self)
        try:
            ws.handshake()
        except Exception:
            return
        session = None

        def safe_event(obj):
            try:
                ws.send_json(obj)
            except Exception:
                pass

        def safe_audio(data):
            try:
                ws.send_bytes(b"\x01" + data)
            except Exception:
                pass

        try:
            ws.send_json({"type": "hello", "status": _status(),
                          "audio": {"sample_rate": cfg["audio"]["sample_rate"],
                                     "tts_rate": 24000}})
            while True:
                kind, payload = ws.recv()
                if kind == "close":
                    break
                if kind == "binary":
                    if session is not None:
                        session.feed(payload)
                    continue
                try:
                    msg = json.loads(payload)
                except Exception:
                    continue
                cmd = msg.get("type")
                if cmd == "start":
                    with STATE["lock"]:
                        if STATE["session"] is not None:
                            ws.send_json({"type": "error", "message": "busy"})
                            continue
                        if getattr(STATE["components"], "tts_custom_loading", False):
                            ws.send_json({"type": "error", "message": "voice loading"})
                            continue
                        session = Session(cfg, STATE["components"],
                                          send_event=safe_event,
                                          send_audio=safe_audio,
                                          mock=STATE["mock"])
                        STATE["session"] = session
                    session.start()
                    ws.send_json({"type": "status", "status": _status()})
                elif cmd == "stop":
                    if session is not None:
                        session.stop()
                        with STATE["lock"]:
                            if STATE["session"] is session:
                                STATE["session"] = None
                        session = None
                elif cmd == "cancel_tts":
                    if session is not None:
                        session.cancel_tts()
                elif cmd == "ping":
                    ws.send_json({"type": "pong", "ts": msg.get("ts")})
                elif cmd == "status":
                    ws.send_json({"type": "status", "status": _status()})
                elif cmd == "set_direction":
                    cfg["llm"]["direction"] = msg.get("direction", "auto")
                    ws.send_json({"type": "direction", "direction": cfg["llm"]["direction"]})
                elif cmd == "set_barge_in":
                    cfg["runtime"]["barge_in"] = bool(msg.get("value", False))
                    ws.send_json({"type": "barge_in", "value": cfg["runtime"]["barge_in"]})
                elif cmd == "set_echo_guard":
                    if session is not None:
                        session.echo_guard = bool(msg.get("value", True))
                    ws.send_json({"type": "echo_guard", "value": bool(msg.get("value", True))})
                elif cmd == "set_mt_backend":
                    c = STATE["components"]
                    backend = c.set_mt_backend(msg.get("value", "opus"))
                    ws.send_json({"type": "mt_backend", "value": backend,
                                  "status": _status()})
                elif cmd == "set_asr_mode":
                    c = STATE["components"]
                    mode = c.set_asr_mode(msg.get("value", "hybrid"))
                    ws.send_json({"type": "asr_mode", "value": mode,
                                  "status": _status()})
                elif cmd == "set_tts_enabled":
                    c = STATE["components"]
                    enabled = bool(msg.get("value", True))
                    c.set_tts_enabled(enabled)
                    if not enabled and session is not None:
                        session.cancel_tts()
                    ws.send_json({"type": "tts_enabled", "value": enabled,
                                  "status": _status()})
                elif cmd == "set_tts_continuous":
                    c = STATE["components"]
                    value = bool(msg.get("value", True))
                    c.set_tts_continuous(value)
                    ws.send_json({"type": "tts_continuous", "value": value,
                                  "status": _status()})
                elif cmd == "set_tts_voice":
                    c = STATE["components"]
                    voice = c.set_tts_voice(msg.get("value", "auto"))
                    ws.send_json({"type": "tts_voice", "value": voice,
                                  "status": _status()})
        except WsClosed:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            if session is not None:
                try:
                    session.stop()
                except Exception:
                    traceback.print_exc()
            with STATE["lock"]:
                if STATE["session"] is session:
                    STATE["session"] = None
            try:
                ws.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--mock", action="store_true",
                    help="no NPU models: energy VAD + canned ASR/translation + tone TTS")
    ap.add_argument("--no-preload", action="store_true")
    ap.add_argument("--tls-port", type=int, default=0,
                    help="also serve HTTPS on this port (needed for LAN microphone access)")
    ap.add_argument("--tls-cert", default="", help="PEM certificate for --tls-port")
    ap.add_argument("--tls-key", default="", help="PEM private key for --tls-port")
    args = ap.parse_args()

    cfg = config_mod.load(args.config)
    if args.host:
        cfg["server"]["host"] = args.host
    if args.port:
        cfg["server"]["port"] = args.port
    STATE["cfg"] = cfg
    STATE["mock"] = args.mock
    print(f"[cfg] {args.config}", flush=True)
    print(f"[cfg] mock={args.mock} work={cfg['runtime']['work_dir']}", flush=True)

    comps = Components(cfg, mock=args.mock)
    STATE["components"] = comps
    if not args.no_preload:
        # Load synchronously: the VNPU mode is device-global, so the LLM must
        # initialise AX Engine before VAD/ASR/TTS join it.  A session starting
        # mid-load could otherwise freeze the board into DISABLE mode.
        comps.preload()
    else:
        print("[cfg] preload disabled", flush=True)

    httpd = ThreadingHTTPServer((cfg["server"]["host"], cfg["server"]["port"]), Handler)
    httpd.daemon_threads = True
    httpd.allow_reuse_address = True

    tlsd = None
    if args.tls_port and args.tls_cert and args.tls_key:
        try:
            import ssl
            tlsd = ThreadingHTTPServer((cfg["server"]["host"], args.tls_port), Handler)
            tlsd.daemon_threads = True
            tlsd.allow_reuse_address = True
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(args.tls_cert, args.tls_key)
            tlsd.socket = ctx.wrap_socket(tlsd.socket, server_side=True)
            threading.Thread(target=tlsd.serve_forever, daemon=True,
                             name="https").start()
            STATE["tls_port"] = args.tls_port
            print(f"[srv] https://{cfg['server']['host']}:{args.tls_port}/ "
                  f"(self-signed; accept the browser warning)", flush=True)
        except Exception:
            traceback.print_exc()
            print("[srv] HTTPS disabled (certificate unreadable?)", flush=True)
            tlsd = None

    def shutdown(*_):
        print("[srv] shutting down", flush=True)
        try:
            if STATE["session"] is not None:
                STATE["session"].stop()
        except Exception:
            pass
        comps.close()
        if tlsd is not None:
            tlsd.shutdown()
        httpd.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"[srv] http://{cfg['server']['host']}:{cfg['server']['port']}/", flush=True)
    try:
        httpd.serve_forever()
    finally:
        comps.close()


if __name__ == "__main__":
    main()
