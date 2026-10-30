"""基础 DSP 组件：状态化滤波器、包络、立体声联动限幅、软削波。

设计原则
--------
1. **所有增益都作用于 L/R 两个声道**（stereo-linked）。任何逐声道独立的压缩/
   限幅都会改变 ILD，直接破坏 HRTF 的左右定位。
2. IIR 滤波器保留 ``zi`` 状态，块与块之间不重置，避免每次 block 边界的“咔哒”。
3. 不引入额外 lookahead（除了限幅器的 0.2 ms 攻击），保证低延迟。
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

EPS = 1e-12


def db_to_lin(db: float | np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=np.float32) / 20.0)


def lin_to_db(x: float | np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float32), EPS))


def design_highpass(sample_rate: int, cutoff_hz: float, order: int = 2) -> np.ndarray:
    return butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")


def design_bandpass(
    sample_rate: int, low_hz: float, high_hz: float, order: int = 2
) -> np.ndarray:
    high_hz = min(high_hz, sample_rate / 2 - 100.0)
    return butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")


class StatefulSOS:
    """多声道 SOS 滤波器，跨 block 保留状态。"""

    def __init__(self, sos: np.ndarray, channels: int = 2):
        self.sos = np.asarray(sos, dtype=np.float64)
        self.channels = channels
        self.zi: np.ndarray | None = None

    def reset(self) -> None:
        self.zi = None

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if self.zi is None:
            # 从零状态开始；不要用 sosfilt_zi(sos)（那是阶跃稳态，
            # 用零输入启动会产生很大的瞬态振铃）。
            self.zi = np.zeros((self.sos.shape[0], x.shape[1], 2), dtype=np.float64)
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return np.asarray(y, dtype=np.float32)


class EnvelopeFollower:
    """单极点包络：上升用 attack，下降用 release。

    输入应为非负值（例如 |x| 或频带 RMS）。逐样本循环是为了支持
    attack/release 不对称；在 128 样本 block 下开销可以忽略。
    """

    def __init__(self, attack_ms: float, release_ms: float, sample_rate: int, initial: float = 0.0):
        self.attack = math.exp(-1.0 / max(0.01, attack_ms * sample_rate / 1000.0))
        self.release = math.exp(-1.0 / max(0.01, release_ms * sample_rate / 1000.0))
        self.value = float(initial)

    def reset(self, initial: float = 0.0) -> None:
        self.value = float(initial)

    def set_times(self, attack_ms: float, release_ms: float, sample_rate: int) -> None:
        self.attack = math.exp(-1.0 / max(0.01, attack_ms * sample_rate / 1000.0))
        self.release = math.exp(-1.0 / max(0.01, release_ms * sample_rate / 1000.0))

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(x.shape[0], dtype=np.float32)
        env = self.value
        a_att, a_rel = self.attack, self.release
        for i in range(x.shape[0]):
            v = float(x[i])
            coef = a_att if v > env else a_rel
            env = coef * env + (1.0 - coef) * v
            out[i] = env
        self.value = env
        return out


def linked_peak(x: np.ndarray) -> np.ndarray:
    """两个声道绝对值的最大值，用于 stereo-linked 检测。"""
    return np.max(np.abs(x), axis=1)


def mid_side(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = x[:, 0]
    right = x[:, 1]
    return 0.5 * (left + right), 0.5 * (left - right)


def from_mid_side(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def apply_side_gain(x: np.ndarray, side_gain: float | np.ndarray) -> np.ndarray:
    """只改变侧信号增益，不动 mid、不引入时延；用于增强左右定位。"""
    mid, side = mid_side(x)
    if np.isscalar(side_gain):
        side = side * float(side_gain)
    else:
        side = side * np.asarray(side_gain, dtype=np.float32)
    return from_mid_side(mid, side)


def soft_clip(x: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    """只在接近 ceiling 时弯曲，正常电平保持线性。"""
    ceiling = float(ceiling)
    if ceiling >= 1.0:
        return np.clip(x, -1.0, 1.0).astype(np.float32)
    a = np.abs(x)
    over = a > ceiling
    if not np.any(over):
        return x.astype(np.float32, copy=False)
    out = x.copy()
    knee = 1.0 - ceiling
    out[over] = np.sign(x[over]) * (
        ceiling + knee * np.tanh((a[over] - ceiling) / max(knee, EPS))
    )
    return out.astype(np.float32)


class LinkedLimiter:
    """零/低 lookahead 的立体声联动限幅器。

    增益由 max(|L|, |R|) 的包络计算，再同时乘到两个声道——因此左右电平差
    （ILD）被保留，方向感不会被压扁。
    """

    def __init__(
        self,
        threshold_db: float,
        attack_ms: float,
        release_ms: float,
        sample_rate: int,
        lookahead_ms: float = 0.0,
    ):
        self.threshold = float(db_to_lin(threshold_db))
        self.env = EnvelopeFollower(attack_ms, release_ms, sample_rate)
        self.lookahead = max(0, int(round(lookahead_ms * sample_rate / 1000.0)))
        self.delay = np.zeros((self.lookahead, 2), dtype=np.float32)

    def reset(self) -> None:
        self.env.reset(0.0)
        self.delay.fill(0.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        peak = linked_peak(x)
        env = self.env.process(peak)
        gain = np.minimum(1.0, self.threshold / np.maximum(env, EPS)).astype(np.float32)

        if self.lookahead == 0:
            return (x * gain[:, None]).astype(np.float32)

        # 音频延迟 lookahead，但增益用“当前”输入计算：这样当延迟后的
        # 峰值到达输出时，增益已经由它之后的样本建立起来（真正的 lookahead）。
        full = np.concatenate([self.delay, x], axis=0)
        out = full[: x.shape[0]].copy()
        self.delay = full[x.shape[0] :]
        return (out * gain[:, None]).astype(np.float32)
