"""增强链：脚步声向上压缩/侧向增强 → 枪声动态高频平滑 → 输出保护。

为什么这样设计
--------------
* 脚步声：用 **向上压缩**（upward compression）抬高低于阈值的微弱瞬态，
  而不是简单把整体音量调大；再用侧信号增益强化左右差异（ILD）。
* 枪声：用 **动态高频衰减 + 立体声联动限幅** 降低刺耳感和峰值，但不对 L/R
  做不同的处理，因此枪声起音的 ITD/ILD 仍然保留。
* 所有处理都是 stereo-linked，避免把双耳线索“压平”。
"""

from __future__ import annotations

import math

import numpy as np

from .analysis import Detection, EventDetector
from .config import FootstepConfig, GunshotConfig, OutputConfig, PipelineConfig
from .dsp import (
    LinkedLimiter,
    StatefulSOS,
    apply_side_gain,
    db_to_lin,
    design_highpass,
    soft_clip,
)


class FootstepEnhancer:
    def __init__(self, cfg: FootstepConfig, sample_rate: int):
        self.cfg = cfg
        self.sr = sample_rate

    def process(self, x: np.ndarray, det: Detection) -> np.ndarray:
        # 枪声期间基本不放大脚步，避免枪声被“脚步化”
        activity = det.foot_activity * (1.0 - 0.85 * det.gun_activity)

        # 向上压缩：低于 upward_threshold_db 的部分按 ratio 抬升
        env_db = det.foot_env_db
        upward_db = (self.cfg.upward_threshold_db - env_db) * (1.0 - 1.0 / self.cfg.upward_ratio)
        upward_db = np.clip(upward_db, 0.0, self.cfg.max_boost_db)
        boost_db = upward_db * activity
        gain = db_to_lin(boost_db)[:, None]

        # 把增强后的脚步频带加回去
        y = x + det.foot_band * (gain - 1.0)

        # 4–9 kHz 细节：轻微提亮鞋底摩擦/耳廓线索，只在脚步活动时生效
        detail_db = self.cfg.detail_boost_db * activity
        detail_gain = db_to_lin(detail_db)[:, None]
        y = y + det.high_band * (detail_gain - 1.0)

        # 侧信号增强：只改变 L-R，不改变 mid，也不引入时延
        side_db = self.cfg.side_boost_db * activity
        side_gain = db_to_lin(side_db)
        y = apply_side_gain(y, side_gain)
        return y.astype(np.float32)


class GunshotSmoother:
    def __init__(self, cfg: GunshotConfig, sample_rate: int):
        self.cfg = cfg
        self.sr = sample_rate
        self.limiter = LinkedLimiter(
            cfg.limiter_threshold_db,
            cfg.limiter_attack_ms,
            cfg.limiter_release_ms,
            sample_rate,
            cfg.limiter_lookahead_ms,
        )

    def reset(self) -> None:
        self.limiter.reset()

    def refresh(self) -> None:
        self.limiter.threshold = float(db_to_lin(self.cfg.limiter_threshold_db))
        self.limiter.env.set_times(
            self.cfg.limiter_attack_ms, self.cfg.limiter_release_ms, self.sr
        )

    def process(self, x: np.ndarray, det: Detection) -> np.ndarray:
        activity = det.gun_activity

        # 动态高频衰减：只切高频带，起音攻击 1.5 ms，释放 180 ms
        cut_db = self.cfg.hf_cut_db * activity
        cut_gain = db_to_lin(cut_db)[:, None]
        y = x + det.high_band * (cut_gain - 1.0)

        # 联动限幅：保护输出，同时保留 L/R 电平差
        y = self.limiter.process(y)
        return y.astype(np.float32)


class OutputStage:
    def __init__(self, cfg: OutputConfig, sample_rate: int):
        self.cfg = cfg
        self.highpass = StatefulSOS(design_highpass(sample_rate, cfg.highpass_hz), channels=2)
        self.ceiling = float(db_to_lin(cfg.ceiling_db))
        self.makeup = float(db_to_lin(cfg.makeup_db))

    def reset(self) -> None:
        self.highpass.reset()

    def process(self, x: np.ndarray) -> np.ndarray:
        y = self.highpass.process(x)
        y = y * self.makeup
        y = soft_clip(y, ceiling=self.ceiling)
        return y.astype(np.float32)


class BinauralPipeline:
    """完整链路：输入增益 → 检测 → 脚步增强 → 枪声平滑 → 输出保护。"""

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        sr = self.cfg.audio.sample_rate
        self.detector = EventDetector(sr, self.cfg.footstep, self.cfg.gunshot)
        self.footstep = FootstepEnhancer(self.cfg.footstep, sr)
        self.gunshot = GunshotSmoother(self.cfg.gunshot, sr)
        self.output = OutputStage(self.cfg.output, sr)
        self.input_gain = float(db_to_lin(self.cfg.audio.input_gain_db))
        self.output_gain = float(db_to_lin(self.cfg.audio.output_gain_db))
        self.bypass = bool(self.cfg.bypass)
        self.last_detection: Detection | None = None

        # 模型检测器（可选）：后台线程给出 foot/gun 概率，这里做时间平滑后
        # 与启发式活动度混合。model_weight=1 表示完全由模型接管。
        self.model_enabled = False
        self.model_weight = 0.8
        self._model_target = (0.0, 0.0)
        self._model_foot = 0.0
        self._model_gun = 0.0

    def set_model_probs(self, foot: float, gun: float, enabled: bool = True) -> None:
        self.model_enabled = bool(enabled)
        if enabled:
            self._model_target = (float(foot), float(gun))

    def refresh_gains(self) -> None:
        """运行中修改增益/上限后调用（不重建滤波器）。"""
        self.input_gain = float(db_to_lin(self.cfg.audio.input_gain_db))
        self.output_gain = float(db_to_lin(self.cfg.audio.output_gain_db))
        self.output.ceiling = float(db_to_lin(self.cfg.output.ceiling_db))
        self.output.makeup = float(db_to_lin(self.cfg.output.makeup_db))
        self.gunshot.refresh()
        self.bypass = bool(self.cfg.bypass)

    def reset(self) -> None:
        self.detector.reset()
        self.gunshot.reset()
        self.output.reset()
        self.last_detection = None

    def process(self, block: np.ndarray) -> np.ndarray:
        x = np.asarray(block, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"需要 [N, 2] 立体声，得到 {x.shape}")
        if x.shape[0] == 0:
            return x

        x = x * self.input_gain
        if self.bypass:
            return soft_clip(x * self.output_gain, ceiling=float(db_to_lin(self.cfg.output.ceiling_db)))

        det = self.detector.process(x)
        if self.model_enabled:
            frames = x.shape[0]
            sr = self.cfg.audio.sample_rate
            attack = 1.0 - math.exp(-frames / max(1.0, 0.08 * sr))
            release = 1.0 - math.exp(-frames / max(1.0, 0.35 * sr))
            tf = max(0.0, (self._model_target[0] - 0.40) / 0.60)
            tg = max(0.0, (self._model_target[1] - 0.40) / 0.60)
            cf = attack if tf > self._model_foot else release
            cg = attack if tg > self._model_gun else release
            self._model_foot += cf * (tf - self._model_foot)
            self._model_gun += cg * (tg - self._model_gun)
            w = self.model_weight
            det.foot_activity = (1.0 - w) * det.foot_activity + w * self._model_foot
            det.gun_activity = (1.0 - w) * det.gun_activity + w * self._model_gun
        self.last_detection = det

        y = self.footstep.process(x, det) if self.cfg.footstep.enabled else x
        y = self.gunshot.process(y, det) if self.cfg.gunshot.enabled else y
        y = self.output.process(y * self.output_gain)
        return y
