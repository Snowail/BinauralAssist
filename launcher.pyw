"""一键启动器（双击 .lnk / .bat 使用）。

* 先检查 http://127.0.0.1:8765 是否已经在运行；在运行就直接退出。
* 否则用 pythonw 启动 binaural_aed.webui（监听 0.0.0.0，手机可通过内网访问）。
* 没有控制台窗口；启动失败会弹出 Windows 消息框。

手机访问：http://<本机内网 IP>:8765
安全：默认无密码；如需密码，把下面 PASSWORD 改成你的密码（会启用 HTTP Basic）。
"""

from __future__ import annotations

import base64
import ctypes
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("BA_PORT", "8765"))
HOST = "0.0.0.0"
PASSWORD = ""  # 例如 "1234"；留空表示无密码


def _url() -> str:
    return f"http://127.0.0.1:{PORT}/"


def _lan_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return None


def _health() -> bool:
    req = urllib.request.Request(_url() + "api/status")
    if PASSWORD:
        token = base64.b64encode(f"user:{PASSWORD}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=1.0) as r:
            return r.status == 200
    except Exception:
        return False


def _message(text: str, title: str = "Binaural Assist") -> None:
    try:
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)
    except Exception:
        print(text)


def main() -> int:
    if _health():
        return 0

    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = ROOT / "python" / "python.exe"
    if not python.exists():
        python = Path(sys.executable)
    cmd = [
        str(python), "-m", "binaural_aed.webui",
        "--host", HOST, "--port", str(PORT),
    ]
    if PASSWORD:
        cmd += ["--password", PASSWORD]
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        # CREATE_NO_WINDOW 隐藏控制台；把 stdout/stderr 落到 webui.log，
        # 这样 pythonw / 无控制台启动时 Flask 不会因 None 输出流退出。
        log = open(ROOT / "webui.log", "ab", buffering=0)
        subprocess.Popen(
            cmd, cwd=str(ROOT), creationflags=flags, close_fds=True,
            stdout=log, stderr=log,
        )
    except Exception as exc:
        _message(f"启动失败：{exc}")
        return 1

    for _ in range(60):
        if _health():
            ip = _lan_ip()
            if ip:
                try:
                    (ROOT / "lan_url.txt").write_text(f"http://{ip}:{PORT}", encoding="utf-8")
                except Exception:
                    pass
            return 0
        time.sleep(0.4)

    ip = _lan_ip()
    extra = f"\n\n手机访问：http://{ip}:{PORT}" if ip else ""
    _message(
        "服务启动超时。\n\n"
        "可能是端口被占用，或 Python 环境不完整。\n"
        "可在项目目录手动运行：\n"
        ".venv\\Scripts\\python -m binaural_aed.webui --port 8765"
        + extra
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
