"""checkpoint 保存/加载：模型结构、标签映射、梅尔归一化统计量一起保存。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import AudioConfig, DIRECTION_CLASSES, EVENT_CLASSES
from .model import ModelConfig, MultiTaskAudioModel


def save_checkpoint(
    path: str | Path,
    model: MultiTaskAudioModel,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": asdict(model.cfg),
        "event_classes": list(EVENT_CLASSES),
        "direction_classes": list(DIRECTION_CLASSES),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[MultiTaskAudioModel, dict[str, Any]]:
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["model_config"])
    audio = cfg_dict.pop("audio", {})
    cfg = ModelConfig(audio=AudioConfig(**audio), **{k: tuple(v) if k == "cnn_widths" else v for k, v in cfg_dict.items()})
    model = MultiTaskAudioModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


__all__ = ["save_checkpoint", "load_model"]
