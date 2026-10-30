"""线程安全的立体声环形缓冲（训练素材回看用）。"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int, channels: int = 2):
        self.capacity = int(capacity)
        self.channels = channels
        self._buf = np.zeros((self.capacity, channels), dtype=np.float32)
        self._write = 0
        self._count = 0
        self._lock = threading.Lock()

    def push(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        n = block.shape[0]
        if n > self.capacity:
            block = block[-self.capacity :]
            n = self.capacity
        with self._lock:
            end = self._write + n
            if end <= self.capacity:
                self._buf[self._write : end] = block
            else:
                first = self.capacity - self._write
                self._buf[self._write :] = block[:first]
                self._buf[: end - self.capacity] = block[first:]
            self._write = end % self.capacity
            self._count = min(self.capacity, self._count + n)

    def latest(self, n: int) -> np.ndarray:
        n = min(int(n), self._count)
        if n <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)
        with self._lock:
            start = (self._write - n) % self.capacity
            if start + n <= self.capacity:
                return self._buf[start : start + n].copy()
            first = self.capacity - start
            return np.concatenate([self._buf[start:], self._buf[: n - first]], axis=0)

    def clear(self) -> None:
        with self._lock:
            self._write = 0
            self._count = 0
