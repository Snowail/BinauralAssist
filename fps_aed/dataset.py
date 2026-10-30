"""数据集与合成数据（用于没有真实录音时跑通训练/推理链路）。

真实数据格式（manifest.csv）
---------------------------
path,event,direction,split
clips/session01/gunshot_0001.wav,gunshot,front_left_level,train
clips/session01/footstep_0007.wav,footstep,back_right_level,val

* path 相对于 manifest 所在目录（或 --root 指定的目录）
* event: background / gunshot / footstep / other
* direction: 12 类之一；background/other 可以留空
* split: train / val / test（可选；不填则用 --val-ratio 随机划分）
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .config import (
    CHANNEL_LAYOUTS,
    DIRECTION_CLASSES,
    DIRECTION_TO_IDX,
    EVENT_CLASSES,
    EVENT_TO_IDX,
    AudioConfig,
    direction_weights_vector,
)


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from scipy.signal import resample_poly

    g = math.gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    return resample_poly(x, up, down, axis=0).astype(np.float32)


def _fit_channels(x: np.ndarray, target: int) -> np.ndarray:
    """把 [N, C] 调整到目标声道数（只做保守的补零/截断，不做上混）。"""
    n, c = x.shape
    if c == target:
        return x
    if c == 6 and target == 8:
        out = np.zeros((n, 8), dtype=np.float32)
        out[:, :6] = x
        return out
    if c > target:
        return x[:, :target]
    out = np.zeros((n, target), dtype=np.float32)
    out[:, :c] = x
    return out


def load_multichannel(
    path: str | Path,
    target_channels: int = 8,
    target_sr: int = 48_000,
) -> np.ndarray:
    """读取任意声道 WAV，返回 [N, target_channels] float32。"""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = _resample(data, sr, target_sr)
    data = _fit_channels(data, target_channels)
    return np.ascontiguousarray(data, dtype=np.float32)


class GameAudioDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        root: str | Path | None = None,
        cfg: AudioConfig | None = None,
        train: bool = True,
        augment: bool | None = None,
        split: str | None = None,
        val_ratio: float = 0.2,
        seed: int = 0,
    ):
        self.cfg = cfg or AudioConfig()
        self.manifest = Path(manifest)
        self.root = Path(root) if root is not None else self.manifest.parent
        self.train = train
        self.augment = train if augment is None else augment

        rows: list[dict] = []
        with self.manifest.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        if not rows:
            raise ValueError(f"manifest 为空: {self.manifest}")

        has_split = any((row.get("split") or "").strip() for row in rows)
        if split is not None and has_split:
            rows = [r for r in rows if (r.get("split") or "").strip() == split]
        elif split is None and not has_split:
            rng = random.Random(seed)
            rng.shuffle(rows)
            n_val = int(round(len(rows) * val_ratio))
            rows = rows[n_val:] if train else rows[:n_val]

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _crop(self, audio: np.ndarray) -> np.ndarray:
        n = self.cfg.segment_samples
        if audio.shape[0] >= n:
            if self.train:
                start = random.randint(0, audio.shape[0] - n)
            else:
                start = (audio.shape[0] - n) // 2
            return audio[start : start + n]
        pad = n - audio.shape[0]
        if self.train:
            left = random.randint(0, pad)
        else:
            left = pad // 2
        return np.pad(audio, ((left, pad - left), (0, 0)), mode="constant")

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.root / path
        audio = load_multichannel(path, self.cfg.channels, self.cfg.sample_rate)
        audio = self._crop(audio)

        event_name = (row.get("event") or "background").strip()
        direction_name = (row.get("direction") or "").strip()
        if event_name not in EVENT_TO_IDX:
            raise ValueError(f"{path}: 未知 event={event_name!r}")
        event = EVENT_TO_IDX[event_name]
        direction = DIRECTION_TO_IDX.get(direction_name, -1)

        x = torch.from_numpy(audio.T.copy())  # [C, N]

        if self.augment:
            x = self._augment(x)

        return {
            "waveform": x.contiguous(),
            "event": torch.tensor(event, dtype=torch.long),
            "direction": torch.tensor(direction, dtype=torch.long),
            "path": str(path),
        }

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # 随机整体增益
        x = x * (10.0 ** (random.uniform(-6.0, 6.0) / 20.0))
        # 轻微声道增益抖动（保留方向线索，只模拟摆放/设备差异）
        x = x * torch.tensor(
            [10.0 ** (random.uniform(-1.5, 1.5) / 20.0) for _ in range(x.shape[0])]
        ).unsqueeze(-1)
        # 白噪声，随机 SNR
        if random.random() < 0.5:
            snr_db = random.uniform(15.0, 40.0)
            sig_power = x.square().mean().clamp_min(1e-8)
            noise_power = sig_power / (10.0 ** (snr_db / 10.0))
            x = x + torch.randn_like(x) * noise_power.sqrt()
        return x.clamp(-1.0, 1.0)


class SyntheticGameAudioDataset(Dataset):
    """纯合成数据：只为跑通维度/训练循环，不代表真实分布。

    枪声 = 带限噪声爆发；脚步 = 低频衰减振荡 + 点击；其他 = 扫频音。
    方向通过 ``direction_channel_weights`` 分配到各声道。
    """

    def __init__(
        self,
        length: int = 512,
        cfg: AudioConfig | None = None,
        train: bool = True,
        seed: int | None = None,
    ):
        self.cfg = cfg or AudioConfig()
        self.length = length
        self.train = train
        if seed is not None:
            torch.manual_seed(seed)
        self._weights = {
            name: torch.tensor(direction_weights_vector(name), dtype=torch.float32)
            for name in DIRECTION_CLASSES
        }

    def __len__(self) -> int:
        return self.length

    def _event_signal(self, kind: int, n: int, rng: torch.Generator) -> torch.Tensor:
        t = torch.arange(n, dtype=torch.float32) / self.cfg.sample_rate
        if kind == 1:  # gunshot
            length = int(0.18 * self.cfg.sample_rate)
            env = torch.exp(-t[:length] / 0.035)
            burst = torch.randn(length, generator=rng) * env
            burst = torch.diff(burst, prepend=burst[:1])  # 粗糙高通
            sig = torch.zeros(n)
            sig[:length] = burst * 0.8
            return sig
        if kind == 2:  # footstep
            length = int(0.09 * self.cfg.sample_rate)
            env = torch.exp(-t[:length] / 0.02)
            thump = torch.sin(2 * math.pi * 90.0 * t[:length]) * env
            click = torch.randn(length, generator=rng) * torch.exp(-t[:length] / 0.004) * 0.4
            sig = torch.zeros(n)
            sig[:length] = (thump + click) * 0.5
            return sig
        # other: 0.3s 扫频
        length = int(0.3 * self.cfg.sample_rate)
        freq = torch.linspace(300.0, 3000.0, length)
        phase = 2 * math.pi * torch.cumsum(freq / self.cfg.sample_rate, dim=0)
        sig = torch.zeros(n)
        sig[:length] = torch.sin(phase) * torch.linspace(0.5, 0.0, length)
        return sig

    def __getitem__(self, index: int) -> dict:
        rng = torch.Generator().manual_seed(index * 7919 + (0 if self.train else 17))
        n = self.cfg.segment_samples
        channels = self.cfg.channels

        event = int(torch.randint(0, len(EVENT_CLASSES), (1,), generator=rng))
        direction = int(torch.randint(0, len(DIRECTION_CLASSES), (1,), generator=rng))
        direction_name = DIRECTION_CLASSES[direction]

        noise = torch.randn(channels, n, generator=rng) * 0.006
        waveform = noise
        if event != 0:
            sig = self._event_signal(event, n, rng)
            start = int(torch.randint(0, n - int(0.3 * self.cfg.sample_rate), (1,), generator=rng))
            end = min(n, start + sig.shape[0])
            weights = self._weights[direction_name].unsqueeze(-1)
            waveform = waveform.clone()
            waveform[:, start:end] += sig[: end - start].unsqueeze(0) * weights

        # 随机整体增益
        waveform = waveform * (10.0 ** (float(torch.empty(1).uniform_(-3, 3, generator=rng)) / 20.0))
        peak = waveform.abs().max().clamp_min(1e-6)
        waveform = waveform / peak * 0.9

        return {
            "waveform": waveform.contiguous(),
            "event": torch.tensor(event, dtype=torch.long),
            "direction": torch.tensor(direction if event in (1, 2) else -1, dtype=torch.long),
            "path": f"synthetic/{index}.wav",
        }


__all__ = [
    "GameAudioDataset",
    "SyntheticGameAudioDataset",
    "load_multichannel",
    "CHANNEL_LAYOUTS",
]
