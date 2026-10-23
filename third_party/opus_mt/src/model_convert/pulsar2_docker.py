"""Minimal Pulsar2 Docker wrapper (copied from Magnetar, MIT/Apache-2.0)."""
import os
import subprocess
from pathlib import Path


def _tail(text, lines=400):
    return "\n".join(text.splitlines()[-lines:])


def run(cmd, timeout=600, log_file=None):
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=timeout)
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        detail = _tail(proc.stdout)
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {' '.join(cmd)}\n{detail}"
            + (f"\n(full log: {log_file})" if log_file else "")
        )
    return _tail(proc.stdout)


def docker_pulsar2(image: str, workspace: str, command: str, timeout=3600,
                   log_file=None) -> str:
    uid, gid = os.getuid(), os.getgid()
    wrapped = (
        "set +e; mkdir -p /root/.hasplm/installed/32434 && "
        "cp -f /root/*.v2c /root/.hasplm/installed/32434/ 2>/dev/null; "
        f"PATH=/usr/local/bin/.venv/bin:/opt/pulsar2:$PATH {command}; "
        f"status=$?; chown -R {uid}:{gid} /workspace; exit $status"
    )
    cmd = ["docker", "run", "--rm", "--network", "host",
           "-v", f"{workspace}:/workspace",
           "-v", "/var/hasplm:/var/hasplm"]
    hasp_src = os.environ.get("PULSAR2_HASP_SRC")
    if hasp_src:
        cmd += ["-v", f"{hasp_src}:/root/.hasplm"]
    cmd += ["-e", "HASP_HOME=/root/.hasplm", image, "-lc", wrapped]
    return run(cmd, timeout=timeout, log_file=log_file)
