"""集中配置：音频参数、7.1 声道布局、事件类别与方向类别。

Windows / WAVE_FORMAT_EXTENSIBLE 的 7.1 声道顺序（按 speaker mask 升序）：
FL, FR, FC, LFE, BL, BR, SL, SR
5.1: FL, FR, FC, LFE, BL, BR
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 音频参数（与需求一致）
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48_000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 112
F_MIN = 20.0
F_MAX = SAMPLE_RATE / 2

# ---------------------------------------------------------------------------
# 声道布局
# ---------------------------------------------------------------------------
CHANNEL_LAYOUTS: dict[int, tuple[str, ...]] = {
    2: ("FL", "FR"),
    6: ("FL", "FR", "FC", "LFE", "BL", "BR"),
    8: ("FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"),
}
DEFAULT_CHANNELS = 8

# 各声道在水平面上的标称角度（度，正=右，负=左，0=正前）
CHANNEL_AZIMUTH_DEG: dict[str, float] = {
    "FL": -30.0,
    "FR": 30.0,
    "FC": 0.0,
    "LFE": 0.0,  # 不参与方向加权
    "BL": -110.0,
    "BR": 110.0,
    "SL": -90.0,
    "SR": 90.0,
}

# ---------------------------------------------------------------------------
# 类别定义
# ---------------------------------------------------------------------------
EVENT_CLASSES: tuple[str, ...] = ("background", "gunshot", "footstep", "other")
EVENT_TO_IDX = {name: i for i, name in enumerate(EVENT_CLASSES)}

AZIMUTH_CLASSES: tuple[str, ...] = (
    "front_left",
    "front_right",
    "back_left",
    "back_right",
)
ELEVATION_CLASSES: tuple[str, ...] = ("up", "level", "down")
AZIMUTH_DEG: dict[str, float] = {
    "front_left": -45.0,
    "front_right": 45.0,
    "back_left": -135.0,
    "back_right": 135.0,
}

# 12 个方向区域：4 个水平方位 × 3 个高度层
DIRECTION_CLASSES: tuple[str, ...] = tuple(
    f"{az}_{el}" for az in AZIMUTH_CLASSES for el in ELEVATION_CLASSES
)
DIRECTION_TO_IDX = {name: i for i, name in enumerate(DIRECTION_CLASSES)}
N_DIRECTIONS = len(DIRECTION_CLASSES)


def split_direction(name: str) -> tuple[str, str]:
    """'front_left_up' -> ('front_left', 'up')"""
    azimuth, elevation = name.rsplit("_", 1)
    if azimuth not in AZIMUTH_DEG or elevation not in ELEVATION_CLASSES:
        raise ValueError(f"未知方向类别: {name}")
    return azimuth, elevation


# ---------------------------------------------------------------------------
# 方向 -> 各声道权重（用于数据合成和实时增强）
# ---------------------------------------------------------------------------
def direction_channel_weights(direction: str, kappa: float = 2.0) -> dict[str, float]:
    """用 von Mises 核给每个声道一个 0..1 的权重。

    只反映水平方位；7.1 没有高度声道，up/down 无法通过声道权重表达。
    返回顺序与 8 声道布局一致（LFE 权重为 0）。
    """
    azimuth, _ = split_direction(direction)
    target = AZIMUTH_DEG[azimuth]
    raw: dict[str, float] = {}
    for ch, angle in CHANNEL_AZIMUTH_DEG.items():
        raw[ch] = 0.0 if ch == "LFE" else math.exp(kappa * math.cos(math.radians(angle - target)))
    peak = max(raw.values())
    return {ch: raw[ch] / peak for ch in CHANNEL_LAYOUTS[DEFAULT_CHANNELS]}


def direction_gain_db(direction: str, max_gain_db: float = 6.0) -> dict[str, float]:
    """方向增强用的 dB 增益：目标方向 +max_gain_db，反方向 -max_gain_db。"""
    weights = direction_channel_weights(direction)
    values = [v for ch, v in weights.items() if ch != "LFE"]
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1.0
    gains: dict[str, float] = {}
    for ch, w in weights.items():
        if ch == "LFE":
            gains[ch] = 0.0
        else:
            gains[ch] = max_gain_db * (2.0 * (w - lo) / span - 1.0)
    return gains


def direction_weights_vector(direction: str) -> list[float]:
    """按 8 声道顺序返回权重（供合成数据直接使用）。"""
    weights = direction_channel_weights(direction)
    return [weights[ch] for ch in CHANNEL_LAYOUTS[DEFAULT_CHANNELS]]


# ---------------------------------------------------------------------------
# 运行配置
# ---------------------------------------------------------------------------
@dataclass
class AudioConfig:
    sample_rate: int = SAMPLE_RATE
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    n_mels: int = N_MELS
    f_min: float = F_MIN
    f_max: float = F_MAX
    channels: int = DEFAULT_CHANNELS
    segment_seconds: float = 1.0

    @property
    def segment_samples(self) -> int:
        return int(round(self.segment_seconds * self.sample_rate))

    def mel_frames(self, samples: int | None = None) -> int:
        n = self.segment_samples if samples is None else samples
        return 1 + n // self.hop_length
