"""把 2ch 事件分类模型接到实时链路。

训练脚本产出 ``runs/train2ch/best.pt``，这里在后台线程里每 200 ms 取最近
1 秒音频跑一次模型，把 footstep / gunshot 概率交给 ``BinauralPipeline``。
模型推理不在 PortAudio 回调里执行，避免阻塞音频线程。
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch

from .buffer import RingBuffer


class ModelEventClassifier:
    def __init__(
        self,
        checkpoint: str | Path,
        sample_rate: int = 48_000,
        window_seconds: float = 1.0,
        interval_seconds: float = 0.2,
    ):
        from fps_aed.checkpoint import load_model
        from fps_aed.config import EVENT_TO_IDX

        self.checkpoint = str(checkpoint)
        self.sample_rate = sample_rate
        self.window = int(window_seconds * sample_rate)
        self.interval = float(interval_seconds)
        self.model, self.meta = load_model(checkpoint, "cpu")
        self.model.eval()
        self.foot_idx = EVENT_TO_IDX["footstep"]
        self.gun_idx = EVENT_TO_IDX["gunshot"]

        self.history: RingBuffer | None = None
        self._probs = (0.0, 0.0)  # (foot, gun)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.runs = 0

    def attach(self, history: RingBuffer) -> None:
        self.history = history

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="model-event-classifier", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def probs(self) -> tuple[float, float]:
        with self._lock:
            return self._probs

    def status(self) -> dict:
        foot, gun = self.probs()
        return {
            "loaded": True,
            "checkpoint": self.checkpoint,
            "foot": foot,
            "gun": gun,
            "runs": self.runs,
            "error": self.error,
        }

    @torch.inference_mode()
    def _loop(self) -> None:
        while not self._stop.is_set():
            hist = self.history
            if hist is not None:
                data = hist.latest(self.window)
                if data.shape[0] >= self.window:
                    try:
                        x = torch.from_numpy(np.ascontiguousarray(data.T[None, :, :]))
                        logits = self.model(x)["event_logits"][0].softmax(-1)
                        with self._lock:
                            self._probs = (float(logits[self.foot_idx]), float(logits[self.gun_idx]))
                        self.runs += 1
                        self.error = None
                    except Exception as exc:
                        self.error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.interval)
