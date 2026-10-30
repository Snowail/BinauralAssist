"""CNN-LSTM 多任务模型：事件分类 + 方向分类。

两种多声道融合方式
------------------
1. ``fusion="stack"``：把 C 个声道的梅尔谱当成 C 通道图像，直接送进 2D CNN
   （early fusion）。实现简单、参数少，但卷积核需要自己学习“第几个通道”
   的位置关系。
2. ``fusion="attention"``：每个声道共享同一个 CNN 编码器，得到每个声道的
   嵌入序列，再把“声道”当成 token 做 cross-channel self-attention
   （late fusion / channel mixing）。对方向任务更友好，因为模型可以显式比较
   声道之间的能量与时序差异，而且换用 5.1/7.1 时只改 token 数即可。

另外可选地拼接 ``SpatialFeatureExtractor`` 的声道能量与 GCC-PHAT 时延。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .config import AudioConfig, N_DIRECTIONS, EVENT_CLASSES
from .features import LogMelFrontend, SpatialFeatureExtractor, SpecAugment, count_parameters


@dataclass
class ModelConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    fusion: str = "attention"  # "attention" | "stack"
    cnn_widths: tuple[int, ...] = (32, 64, 128, 128)
    lstm_hidden: int = 128
    lstm_layers: int = 2
    dropout: float = 0.3
    attention_heads: int = 4
    attention_layers: int = 2
    use_spatial_aux: bool = True
    use_spec_augment: bool = True
    mel_mean: float = 0.0
    mel_std: float = 1.0
    n_events: int = len(EVENT_CLASSES)
    n_directions: int = N_DIRECTIONS


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PerChannelEncoder(nn.Module):
    """共享 CNN：输入 [B*C, 1, M, T] -> [B*C, D, M', T']。"""

    def __init__(self, widths: tuple[int, ...] = (32, 64, 128, 128)):
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = 1
        for i, width in enumerate(widths):
            blocks.append(ConvBlock(in_ch, width, pool=i < len(widths) - 1))
            in_ch = width
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class StackEncoder(nn.Module):
    """把声道维当成输入通道的 2D CNN：输入 [B, C, M, T] -> [B, D, M', T']。"""

    def __init__(self, in_channels: int, widths: tuple[int, ...] = (32, 64, 128, 128)):
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = in_channels
        for i, width in enumerate(widths):
            blocks.append(ConvBlock(in_ch, width, pool=i < len(widths) - 1))
            in_ch = width
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MultiTaskAudioModel(nn.Module):
    """波形 [B, C, N] -> {"event_logits", "direction_logits", "embedding"}。"""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg

        self.frontend = LogMelFrontend(
            c.audio,
            mean=c.mel_mean,
            std=c.mel_std,
        )
        self.spec_augment = SpecAugment() if c.use_spec_augment else None

        if c.fusion == "stack":
            self.encoder: nn.Module = StackEncoder(c.audio.channels, c.cnn_widths)
        elif c.fusion == "attention":
            self.encoder = PerChannelEncoder(c.cnn_widths)
        else:
            raise ValueError(f"未知 fusion: {c.fusion}")

        hidden = c.lstm_hidden
        feat_dim = 2 * hidden

        if c.fusion == "attention":
            dim = self.encoder.out_channels
            self.channel_embedding = nn.Parameter(torch.zeros(1, c.audio.channels, dim))
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=c.attention_heads,
                dim_feedforward=dim * 2,
                dropout=c.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.channel_attention = nn.TransformerEncoder(
                layer, num_layers=c.attention_layers, enable_nested_tensor=False
            )

        self.lstm = nn.LSTM(
            input_size=self.encoder.out_channels,
            hidden_size=hidden,
            num_layers=c.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=c.dropout if c.lstm_layers > 1 else 0.0,
        )
        self.time_score = nn.Linear(feat_dim, 1)

        self.spatial = SpatialFeatureExtractor(c.audio.channels, c.audio.sample_rate) if c.use_spatial_aux else None
        if self.spatial is not None:
            self.aux_proj = nn.Sequential(
                nn.Linear(self.spatial.out_dim, 64),
                nn.ReLU(inplace=True),
            )
            feat_dim += 64

        self.dropout = nn.Dropout(c.dropout)
        self.event_head = nn.Linear(feat_dim, c.n_events)
        self.direction_head = nn.Linear(feat_dim, c.n_directions)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.dim() != 3:
            raise ValueError(f"期望 [B, C, N]，得到 {tuple(waveform.shape)}")
        b, channels, _ = waveform.shape
        if channels != self.cfg.audio.channels:
            raise ValueError(f"期望 {self.cfg.audio.channels} 声道，得到 {channels}")

        mel = self.frontend(waveform)  # [B, C, M, T]
        if self.spec_augment is not None and self.training:
            m = mel.shape[-2]
            t = mel.shape[-1]
            mel = self.spec_augment(mel.reshape(b * channels, m, t)).reshape(b, channels, m, t)

        if self.cfg.fusion == "attention":
            x = mel.reshape(b * channels, 1, mel.shape[-2], mel.shape[-1])
            x = self.encoder(x)  # [B*C, D, F', T']
            d, f, t = x.shape[1], x.shape[2], x.shape[3]
            x = x.mean(dim=2)  # 频率维平均 -> [B*C, D, T']
            x = x.reshape(b, channels, d, t).permute(0, 3, 1, 2)  # [B, T', C, D]
            x = x.reshape(b * t, channels, d) + self.channel_embedding
            x = self.channel_attention(x)  # cross-channel attention
            x = x.mean(dim=1).reshape(b, t, d)
        else:
            x = self.encoder(mel)  # [B, D, F', T']
            x = x.mean(dim=2).permute(0, 2, 1)  # [B, T', D]

        x, _ = self.lstm(x)  # [B, T', 2H]
        weights = torch.softmax(self.time_score(x).squeeze(-1), dim=1)  # [B, T']
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)

        if self.spatial is not None:
            aux = self.spatial(waveform)
            pooled = torch.cat([pooled, self.aux_proj(aux)], dim=-1)

        pooled = self.dropout(pooled)
        return {
            "event_logits": self.event_head(pooled),
            "direction_logits": self.direction_head(pooled),
            "embedding": pooled,
        }

    @torch.inference_mode()
    def predict(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        out = self.forward(waveform)
        self.train(was_training)
        return {
            "event_probs": out["event_logits"].softmax(dim=-1),
            "direction_probs": out["direction_logits"].softmax(dim=-1),
        }

    @property
    def num_parameters(self) -> int:
        return count_parameters(self)


def build_model(**overrides) -> MultiTaskAudioModel:
    cfg = ModelConfig(**overrides)
    return MultiTaskAudioModel(cfg)


__all__ = ["ModelConfig", "MultiTaskAudioModel", "build_model", "PerChannelEncoder", "StackEncoder"]
