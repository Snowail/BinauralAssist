"""后台训练任务：从 UI 启动 ``fps_aed.train``，收集日志，完成后可热加载模型。"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from .training import TrainingStore


class TrainingRunner:
    def __init__(self, root: str | Path, store: TrainingStore, on_success=None):
        self.root = Path(root)
        self.store = store
        self.on_success = on_success
        self.out_dir = self.root / "runs" / "train2ch"
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lines: deque[str] = deque(maxlen=1000)
        self.running = False
        self.returncode: int | None = None
        self.error: str | None = None
        self.device = "cpu"
        self.auto_replace = True
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.checkpoint: str | None = None

    def start(self, device: str = "cpu", epochs: int = 20, batch_size: int = 16, auto_replace: bool = True) -> dict:
        with self._lock:
            if self.running:
                raise RuntimeError("已有训练任务在运行")
            device = (device or "cpu").lower()
            if device not in ("cpu", "cuda", "auto"):
                raise ValueError("device 必须是 cpu / cuda / auto")
            if device == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA 不可用；AMD GPU 请选择 CPU（或自行安装 torch-directml 适配）")
                except ImportError:
                    raise RuntimeError("当前环境没有安装 torch")

            manifest = self.store.export_manifest()
            if manifest["rows"] < 4:
                raise RuntimeError("有效训练片段太少（至少 4 段，建议每类 100+）")

            epochs = max(1, min(200, int(epochs)))
            batch_size = max(1, min(64, int(batch_size)))
            self.out_dir.mkdir(parents=True, exist_ok=True)
            python_exe = Path(sys.executable)
            if python_exe.name.lower() == "pythonw.exe":
                python_exe = python_exe.with_name("python.exe")
            cmd = [
                str(python_exe), "-u", "-m", "fps_aed.train",
                "--manifest", manifest["path"],
                "--channels", "2",
                "--device", device,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--out", str(self.out_dir),
            ]
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            self._lines.clear()
            self._lines.append("$ " + " ".join(cmd))
            self.running = True
            self.returncode = None
            self.error = None
            self.device = device
            self.auto_replace = bool(auto_replace)
            self.started_at = time.time()
            self.finished_at = None
            self.checkpoint = None
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self.root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            self._thread = threading.Thread(target=self._reader, name="training-runner", daemon=True)
            self._thread.start()
            return self.snapshot()

    def _reader(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            self._lines.append(line.rstrip("\r\n"))
        code = proc.wait()
        with self._lock:
            self.running = False
            self.returncode = code
            self.finished_at = time.time()
            best = self.out_dir / "best.pt"
            if code == 0 and best.exists():
                self.checkpoint = str(best)
                self._lines.append(f"checkpoint: {best}")
                if self.auto_replace and self.on_success is not None:
                    try:
                        self.on_success(str(best))
                        self._lines.append("已自动加载到实时检测器")
                    except Exception as exc:
                        self._lines.append(f"自动加载失败: {exc}")
            elif code != 0:
                self.error = f"训练进程退出码 {code}"
                self._lines.append(self.error)

    def stop(self) -> dict:
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                self._lines.append("训练已手动停止")
            self.running = False
        return self.snapshot()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "returncode": self.returncode,
                "error": self.error,
                "device": self.device,
                "auto_replace": self.auto_replace,
                "checkpoint": self.checkpoint,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "log": list(self._lines)[-160:],
            }
