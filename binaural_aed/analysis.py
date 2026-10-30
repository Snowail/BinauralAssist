"""事件检测：基于“快/慢包络比 + 频带能量”的轻量瞬态检测。

不依赖神经网络，因此没有推理延迟，适合 20–50 ms 的低延迟链路。
检测器输出连续的 activity（0..1），而不是硬 0/1，便于后面做平滑增益，
避免门限抖动造成“抽气”。

分类直觉
--------
* 脚步：100–2600 Hz 的短瞬态，高频相对少，起音比枪声慢；
* 枪声：2.5–9 kHz 的强瞬态，起音极快，高频能量高。
阈值可以在 ``config.py`` 里调；首次使用请用 ``--monitor`` 观察。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FootstepConfig, GunshotConfig
from .dsp import (
    EnvelopeFollower,
    StatefulSOS,
    design_bandpass,
    lin_to_db,
    linked_peak,
)


def _sigmoid(x: np.ndarray | float, width: float = 3.0) -> np.ndarray:
    z = np.asarray(x, dtype=np.float32) / max(width, 1e-6)
    return (1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))).astype(np.float32)


@dataclass
class Detection:
    foot_band: np.ndarray       # [N, 2] 带通后的脚步频带
    high_band: np.ndarray       # [N, 2] 带通后的枪声高频带
    foot_activity: np.ndarray   # [N] 0..1
    gun_activity: np.ndarray    # [N] 0..1
    foot_env_db: np.ndarray     # [N]
    high_env_db: np.ndarray     # [N]
    foot_transient_db: np.ndarray
    gun_transient_db: np.ndarray


class EventDetector:
    def __init__(
        self,
        sample_rate: int,
        foot_cfg: FootstepConfig,
        gun_cfg: GunshotConfig,
        channels: int = 2,
    ):
        self.sr = sample_rate
        self.foot_cfg = foot_cfg
        self.gun_cfg = gun_cfg

        self.foot_filter = StatefulSOS(
            design_bandpass(sample_rate, foot_cfg.band_low_hz, foot_cfg.band_high_hz),
            channels,
        )
        self.high_filter = StatefulSOS(
            design_bandpass(sample_rate, gun_cfg.detector_low_hz, gun_cfg.detector_high_hz),
            channels,
        )

        self.foot_fast = EnvelopeFollower(foot_cfg.detector_attack_ms, foot_cfg.detector_release_ms, sample_rate)
        self.foot_slow = EnvelopeFollower(80.0, 400.0, sample_rate)
        self.high_fast = EnvelopeFollower(gun_cfg.detector_attack_ms, gun_cfg.detector_release_ms, sample_rate)
        self.high_slow = EnvelopeFollower(30.0, 300.0, sample_rate)

        # activity 的平滑：起音快、释放慢
        self.foot_activity_follower = EnvelopeFollower(8.0, 120.0, sample_rate)
        self.gun_activity_follower = EnvelopeFollower(2.0, 150.0, sample_rate)

    def reset(self) -> None:
        for obj in (
            self.foot_filter, self.high_filter,
            self.foot_fast, self.foot_slow, self.high_fast, self.high_slow,
            self.foot_activity_follower, self.gun_activity_follower,
        ):
            obj.reset()

    def process(self, x: np.ndarray) -> Detection:
        foot_band = self.foot_filter.process(x)
        high_band = self.high_filter.process(x)

        foot_mag = linked_peak(foot_band)
        high_mag = linked_peak(high_band)

        foot_fast = self.foot_fast.process(foot_mag)
        foot_slow = self.foot_slow.process(foot_mag + 1e-9)
        high_fast = self.high_fast.process(high_mag)
        high_slow = self.high_slow.process(high_mag + 1e-9)

        foot_env_db = lin_to_db(foot_fast)
        high_env_db = lin_to_db(high_fast)
        foot_transient_db = lin_to_db(foot_fast) - lin_to_db(foot_slow)
        gun_transient_db = lin_to_db(high_fast) - lin_to_db(high_slow)

        # --- 枪声：高频能量 + 瞬态 ---
        gun_gate = _sigmoid(high_env_db - self.gun_cfg.detect_threshold_db, 3.0)
        gun_gate *= _sigmoid(gun_transient_db - self.gun_cfg.transient_threshold_db, 2.0)
        gun_activity = self.gun_activity_follower.process(gun_gate)

        # --- 脚步：低频瞬态；枪声期间抑制，避免把枪声当脚步放大 ---
        foot_gate = _sigmoid(foot_env_db - self.foot_cfg.gate_db, 3.0)
        foot_gate *= _sigmoid(
            foot_transient_db - self.foot_cfg.transient_threshold_db, 2.0
        )
        foot_gate *= 1.0 - 0.85 * gun_activity
        foot_activity = self.foot_activity_follower.process(foot_gate)

        return Detection(
            foot_band=foot_band,
            high_band=high_band,
            foot_activity=foot_activity,
            gun_activity=gun_activity,
            foot_env_db=foot_env_db,
            high_env_db=high_env_db,
            foot_transient_db=foot_transient_db,
            gun_transient_db=gun_transient_db,
        )
