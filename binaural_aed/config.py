"""参数配置：音频设备、脚步声增强、枪声平滑、输出级。

所有阈值都是 dBFS（满刻度为 0 dB）。实际游戏电平差异很大，首次使用请用
``--monitor`` 观察 foot_env_db / gun_env_db，再调整本节参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    sample_rate: int = 48_000
    block_size: int = 128          # 128 @48k = 2.67 ms
    channels: int = 2
    input_device: int | None = None
    output_device: int | None = None
    latency: str = "low"
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0


@dataclass
class FootstepConfig:
    enabled: bool = True
    # 脚步主要能量：低频撞击 + 1–3 kHz 的鞋底/地面细节
    band_low_hz: float = 110.0
    band_high_hz: float = 2600.0

    # 检测：快/慢包络比（transient）超过阈值才算“起音”
    transient_threshold_db: float = 5.0
    gate_db: float = -58.0          # 低于此电平不放大，避免拉高底噪
    detector_attack_ms: float = 2.0
    detector_release_ms: float = 60.0

    # 向上压缩：把“低于阈值”的微弱声音抬起来
    upward_threshold_db: float = -32.0
    upward_ratio: float = 2.5
    max_boost_db: float = 14.0
    attack_ms: float = 4.0
    release_ms: float = 140.0

    # 侧信号（L-R）增益：增强左右定位，不动 ITD
    side_boost_db: float = 4.0

    # 4–9 kHz 细节（耳廓/鞋底摩擦）轻微提亮，帮助前后/材质线索
    detail_boost_db: float = 2.0


@dataclass
class GunshotConfig:
    enabled: bool = True
    detector_low_hz: float = 2500.0
    detector_high_hz: float = 9000.0
    detect_threshold_db: float = -30.0
    transient_threshold_db: float = 5.0
    detector_attack_ms: float = 0.5
    detector_release_ms: float = 45.0

    # 动态高频衰减：降低“刺耳”的 2.5–9 kHz，但保留起音方向
    hf_cut_db: float = -9.0
    cut_attack_ms: float = 1.5
    cut_release_ms: float = 180.0

    # 立体声联动限幅器：保护输出，且不破坏左右电平差
    limiter_threshold_db: float = -9.0
    limiter_attack_ms: float = 0.2
    limiter_release_ms: float = 90.0
    limiter_lookahead_ms: float = 1.5


@dataclass
class OutputConfig:
    highpass_hz: float = 30.0
    ceiling_db: float = -1.0
    makeup_db: float = 0.0


@dataclass
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    footstep: FootstepConfig = field(default_factory=FootstepConfig)
    gunshot: GunshotConfig = field(default_factory=GunshotConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    bypass: bool = False
