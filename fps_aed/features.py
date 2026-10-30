"""多声道特征提取：梅尔频谱、SpecAugment、声道间空间特征。

设计要点
--------
* 梅尔滤波器用 HTK 公式在 PyTorch 中直接实现，不依赖 torchaudio/librosa，
  避免实时部署时的额外依赖。
* 归一化使用训练集全局均值/方差（而不是逐样本归一化），否则会抹掉
  声道间电平差（ILD）——而 ILD 正是方向判别最重要的线索之一。
* ``SpatialFeatureExtractor`` 额外提供每个声道的对数能量和若干声道对的
  GCC-PHAT 时延，作为方向分支的辅助输入（可选）。
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import AudioConfig, CHANNEL_LAYOUTS


def hz_to_mel(freq: torch.Tensor | float) -> torch.Tensor | float:
    return 2595.0 * torch.log10(1.0 + torch.as_tensor(freq) / 700.0)


def mel_to_hz(mel: torch.Tensor | float) -> torch.Tensor | float:
    return 700.0 * (10.0 ** (torch.as_tensor(mel) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """返回 [n_mels, n_fft//2+1] 的三角滤波器组（HTK mel scale）。"""
    f_max = sample_rate / 2 if f_max is None else f_max
    all_freqs = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)

    mel_min = float(hz_to_mel(f_min))
    mel_max = float(hz_to_mel(f_max))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fdiff = hz_points[1:] - hz_points[:-1]
    ramps = hz_points.unsqueeze(1) - all_freqs.unsqueeze(0)  # [n_mels+2, F]

    lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
    upper = ramps[2:] / fdiff[1:].unsqueeze(1)
    fb = torch.clamp(torch.minimum(lower, upper), min=0.0)

    # 避免出现全零滤波器（极端参数下）
    fb = torch.where(fb.sum(dim=1, keepdim=True) > 0, fb, torch.full_like(fb, 1.0 / fb.shape[1]))
    return fb


class LogMelSpectrogram(nn.Module):
    """[..., N] 波形 -> [..., n_mels, T] 对数梅尔谱（dB）。"""

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("window", torch.hann_window(cfg.n_fft), persistent=False)
        self.register_buffer(
            "filterbank",
            mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.f_min, cfg.f_max),
            persistent=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() < 2:
            raise ValueError("waveform 至少是 2D: [..., samples]")
        lead_shape = waveform.shape[:-1]
        flat = waveform.reshape(-1, waveform.shape[-1])

        spec = torch.stft(
            flat,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.real.square() + spec.imag.square()  # [B, F, T]
        mel = torch.einsum("mf,bft->bmt", self.filterbank, power)
        logmel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        return logmel.reshape(*lead_shape, logmel.shape[-2], logmel.shape[-1])


class LogMelFrontend(nn.Module):
    """对数梅尔 + 数据集级归一化（mean/std 为标量或 [n_mels, 1]）。

    注意：这里的归一化统计量在训练集上一次性算好并写进 checkpoint，
    推理时必须使用同一组统计量。
    """

    def __init__(
        self,
        cfg: AudioConfig,
        mean: float | torch.Tensor | None = None,
        std: float | torch.Tensor | None = None,
    ):
        super().__init__()
        self.mel = LogMelSpectrogram(cfg)
        if mean is None:
            mean = 0.0
        if std is None:
            std = 1.0
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        std_t = torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("mean", mean_t.reshape(-1, 1) if mean_t.numel() > 1 else mean_t.reshape(1, 1))
        self.register_buffer("std", std_t.reshape(-1, 1) if std_t.numel() > 1 else std_t.reshape(1, 1))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, C, N] -> [B, C, M, T]
        mel = self.mel(waveform)
        return (mel - self.mean) / (self.std + 1e-6)


class SpecAugment(nn.Module):
    """频带/时间遮挡。只在训练时启用。"""

    def __init__(
        self,
        freq_masks: int = 2,
        freq_width: int = 8,
        time_masks: int = 2,
        time_width: int = 8,
    ):
        super().__init__()
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.time_masks = time_masks
        self.time_width = time_width

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, M, T]
        if not self.training or (self.freq_masks == 0 and self.time_masks == 0):
            return mel
        b, m, t = mel.shape
        mask = mel.new_ones((b, m, t))
        for i in range(b):
            for _ in range(self.freq_masks):
                width = int(torch.randint(0, self.freq_width + 1, (1,)))
                if width == 0 or width >= m:
                    continue
                start = int(torch.randint(0, m - width, (1,)))
                mask[i, start : start + width, :] = 0.0
            for _ in range(self.time_masks):
                width = int(torch.randint(0, self.time_width + 1, (1,)))
                if width == 0 or width >= t:
                    continue
                start = int(torch.randint(0, t - width, (1,)))
                mask[i, :, start : start + width] = 0.0
        return mel * mask


# ---------------------------------------------------------------------------
# 声道间空间特征
# ---------------------------------------------------------------------------
DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("FL", "FR"),
    ("FL", "BL"),
    ("FR", "BR"),
    ("BL", "BR"),
    ("FL", "SL"),
    ("FR", "SR"),
)


class SpatialFeatureExtractor(nn.Module):
    """每个声道的对数 RMS + 若干声道对的 GCC-PHAT 时延。

    输出维度 = C + 2 * len(pairs)（时延归一化值、归一化互相关峰值）。
    """

    def __init__(
        self,
        channels: int = 8,
        sample_rate: int = 48_000,
        pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS,
        max_delay_ms: float = 1.0,
        fft_size: int = 8192,
    ):
        super().__init__()
        layout = CHANNEL_LAYOUTS[channels]
        index = {name: i for i, name in enumerate(layout)}
        self.pairs = [(index[a], index[b]) for a, b in pairs if a in index and b in index]
        self.max_lag = max(1, int(round(sample_rate * max_delay_ms / 1000.0)))
        self.fft_size = fft_size
        self.n_channels = len(layout)
        self.register_buffer("_window", torch.hann_window(fft_size), persistent=False)

    @property
    def out_dim(self) -> int:
        return self.n_channels + 2 * len(self.pairs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, C, N]
        b, c, n = waveform.shape
        eps = 1e-8
        rms = torch.sqrt(waveform.square().mean(dim=-1) + eps)
        log_rms = 20.0 * torch.log10(rms + eps)
        log_rms = log_rms - log_rms.mean(dim=1, keepdim=True)  # 只保留相对电平

        if not self.pairs:
            return log_rms

        # 取中间最多 fft_size 个采样点，加窗后做 GCC-PHAT
        seg = waveform
        if n > self.fft_size:
            start = (n - self.fft_size) // 2
            seg = seg[..., start : start + self.fft_size]
        elif n < self.fft_size:
            seg = torch.nn.functional.pad(seg, (0, self.fft_size - n))
        seg = seg * self._window

        feats: list[torch.Tensor] = []
        nfft = self.fft_size
        for i, j in self.pairs:
            a = torch.fft.rfft(seg[:, i, :], n=nfft, dim=-1)
            bspec = torch.fft.rfft(seg[:, j, :], n=nfft, dim=-1)
            cross = a * torch.conj(bspec)
            cross = cross / (cross.abs() + eps)  # PHAT 加权
            cc = torch.fft.irfft(cross, n=nfft, dim=-1)

            # 正时延在开头，负时延在结尾：拼成 [-max_lag, +max_lag]
            pos = cc[..., : self.max_lag + 1]
            neg = cc[..., -self.max_lag :]
            candidate = torch.cat([pos, neg], dim=-1)  # [B, 2*max_lag+1]

            peak_idx = candidate.argmax(dim=-1)
            lags = torch.cat(
                [
                    torch.arange(0, self.max_lag + 1, device=waveform.device),
                    torch.arange(-self.max_lag, 0, device=waveform.device),
                ]
            ).float()
            lag = lags[peak_idx] / self.max_lag
            peak = candidate.gather(-1, peak_idx.unsqueeze(-1)).squeeze(-1)
            feats.extend([lag, peak])

        return torch.cat([log_rms] + [f.unsqueeze(-1) for f in feats], dim=-1)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


__all__ = [
    "LogMelSpectrogram",
    "LogMelFrontend",
    "SpecAugment",
    "SpatialFeatureExtractor",
    "mel_filterbank",
    "hz_to_mel",
    "mel_to_hz",
    "count_parameters",
]
