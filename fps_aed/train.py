"""训练入口：多任务（事件 + 方向）CNN-LSTM。

快速验证（不需要数据、不需要声卡）::

    python -m fps_aed.train --synthetic --smoke

真实数据::

    python -m fps_aed.train --manifest data/manifest.csv --epochs 30 \
        --batch-size 16 --out runs/baseline
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import (
    DIRECTION_CLASSES,
    EVENT_CLASSES,
    EVENT_TO_IDX,
    AudioConfig,
)
from .dataset import GameAudioDataset, SyntheticGameAudioDataset
from .model import ModelConfig, MultiTaskAudioModel
from .checkpoint import save_checkpoint

GUNSHOT = EVENT_TO_IDX["gunshot"]
FOOTSTEP = EVENT_TO_IDX["footstep"]
DIRECTION_EVENTS = (GUNSHOT, FOOTSTEP)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def macro_f1(cm: np.ndarray) -> tuple[float, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean()), f1


@torch.inference_mode()
def evaluate(
    model: MultiTaskAudioModel,
    loader: DataLoader,
    device: torch.device,
    event_weights: torch.Tensor | None = None,
    dir_weight: float = 1.0,
) -> dict:
    model.eval()
    ev_true: list[int] = []
    ev_pred: list[int] = []
    dir_true: list[int] = []
    dir_pred: list[int] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        wav = batch["waveform"].to(device)
        event = batch["event"].to(device)
        direction = batch["direction"].to(device)
        out = model(wav)

        loss = F.cross_entropy(out["event_logits"], event, weight=event_weights)
        mask = ((event == GUNSHOT) | (event == FOOTSTEP)) & (direction >= 0)
        if mask.any():
            loss = loss + dir_weight * F.cross_entropy(
                out["direction_logits"][mask], direction[mask]
            )
        total_loss += float(loss)
        n_batches += 1

        ed = out["event_logits"].argmax(-1)
        ev_true.extend(event.cpu().tolist())
        ev_pred.extend(ed.cpu().tolist())
        if mask.any():
            dd = out["direction_logits"][mask].argmax(-1)
            dir_true.extend(direction[mask].cpu().tolist())
            dir_pred.extend(dd.cpu().tolist())

    if not ev_true:
        return {
            "loss": total_loss / max(n_batches, 1),
            "event_acc": 0.0,
            "event_f1": 0.0,
            "event_cm": np.zeros((len(EVENT_CLASSES), len(EVENT_CLASSES)), dtype=np.int64),
            "dir_acc": 0.0,
            "dir_f1": 0.0,
            "dir_count": 0,
        }

    event_cm = confusion_matrix(np.array(ev_true), np.array(ev_pred), len(EVENT_CLASSES))
    event_f1, _ = macro_f1(event_cm)
    result = {
        "loss": total_loss / max(n_batches, 1),
        "event_acc": float(np.trace(event_cm) / max(event_cm.sum(), 1)),
        "event_f1": event_f1,
        "event_cm": event_cm,
    }
    if dir_true:
        dir_cm = confusion_matrix(np.array(dir_true), np.array(dir_pred), len(DIRECTION_CLASSES))
        dir_f1, _ = macro_f1(dir_cm)
        result.update(
            {
                "dir_acc": float(np.trace(dir_cm) / max(dir_cm.sum(), 1)),
                "dir_f1": dir_f1,
                "dir_cm": dir_cm,
                "dir_count": len(dir_true),
            }
        )
    else:
        result.update({"dir_acc": 0.0, "dir_f1": 0.0, "dir_count": 0})
    return result


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def _manifest_has_split(path: Path) -> bool:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return "split" in (reader.fieldnames or [])


def build_datasets(args, cfg: AudioConfig):
    if args.synthetic:
        train = SyntheticGameAudioDataset(1024, cfg, train=True, seed=args.seed)
        val = SyntheticGameAudioDataset(256, cfg, train=False, seed=args.seed + 1)
        return train, val

    if not args.manifest:
        raise SystemExit("必须提供 --manifest，或使用 --synthetic")
    manifest = Path(args.manifest)
    split_mode = _manifest_has_split(manifest)
    common = dict(manifest=manifest, root=args.root, cfg=cfg, seed=args.seed)
    train = GameAudioDataset(train=True, split="train" if split_mode else None, **common)
    val = GameAudioDataset(
        train=False, augment=False, split="val" if split_mode else None, **common
    )
    return train, val


def class_weights(dataset, n_events: int) -> torch.Tensor:
    counts = np.zeros(n_events, dtype=np.float64)
    if isinstance(dataset, GameAudioDataset):
        for row in dataset.rows:
            name = (row.get("event") or "background").strip()
            if name in EVENT_TO_IDX:
                counts[EVENT_TO_IDX[name]] += 1
    else:
        counts[:] = 1.0
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_events * counts)
    return torch.tensor(weights, dtype=torch.float32)


def load_mel_stats(path: str | None):
    if not path:
        return 0.0, 1.0
    data = np.load(path)
    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)
    return mean, std


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train(args) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用：AMD GPU 请用 --device cpu（或安装 torch-directml 后自行适配）")
    cfg = AudioConfig(segment_seconds=args.segment_seconds, channels=args.channels)
    train_ds, val_ds = build_datasets(args, cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_ds) > args.batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    mel_mean, mel_std = load_mel_stats(args.mel_stats)
    model_cfg = ModelConfig(
        audio=cfg,
        fusion=args.fusion,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        use_spatial_aux=not args.no_spatial_aux,
        use_spec_augment=not args.no_spec_augment,
        mel_mean=mel_mean,
        mel_std=mel_std,
    )
    model = MultiTaskAudioModel(model_cfg).to(device)
    print(f"模型参数量: {model.num_parameters/1e6:.2f}M  device={device}  fusion={args.fusion}")

    weights = class_weights(train_ds, len(EVENT_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, min(len(train_loader), args.max_batches or len(train_loader)))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * 0.05))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "history.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "event_acc", "event_f1", "dir_acc", "dir_f1", "lr", "seconds"]
        )

    best_score = -1.0
    best_path = out_dir / "best.pt"
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running = 0.0
        n_steps = 0

        for step, batch in enumerate(train_loader):
            if args.max_batches and step >= args.max_batches:
                break
            wav = batch["waveform"].to(device, non_blocking=True)
            event = batch["event"].to(device, non_blocking=True)
            direction = batch["direction"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(wav)
                loss = F.cross_entropy(
                    out["event_logits"], event, weight=weights, label_smoothing=args.label_smoothing
                )
                mask = ((event == GUNSHOT) | (event == FOOTSTEP)) & (direction >= 0)
                if mask.any():
                    dir_loss = F.cross_entropy(
                        out["direction_logits"][mask],
                        direction[mask],
                        label_smoothing=args.label_smoothing,
                    )
                    loss = loss + args.dir_weight * dir_loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            scheduler.step()
            global_step += 1
            running += float(loss.detach())
            n_steps += 1

        train_loss = running / max(n_steps, 1)
        metrics = evaluate(model, val_loader, device, weights, args.dir_weight)
        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.4f}  "
            f"val {metrics['loss']:.4f}  ev_acc {metrics['event_acc']:.3f}  "
            f"ev_f1 {metrics['event_f1']:.3f}  dir_acc {metrics['dir_acc']:.3f}  "
            f"dir_f1 {metrics['dir_f1']:.3f}  lr {lr:.2e}  {elapsed:.1f}s"
        )
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch, f"{train_loss:.6f}", f"{metrics['loss']:.6f}",
                    f"{metrics['event_acc']:.6f}", f"{metrics['event_f1']:.6f}",
                    f"{metrics['dir_acc']:.6f}", f"{metrics['dir_f1']:.6f}",
                    f"{lr:.8f}", f"{elapsed:.2f}",
                ]
            )

        # 以事件 F1 + 方向 F1 的均值作为选择指标
        score = 0.5 * (metrics["event_f1"] + metrics["dir_f1"])
        if score >= best_score:
            best_score = score
            save_checkpoint(
                best_path,
                model,
                epoch=epoch,
                metrics={k: float(v) for k, v in metrics.items() if not k.endswith("_cm")},
                optimizer=optimizer,
                extra={"args": vars(args), "event_cm": metrics["event_cm"].tolist(),
                       "dir_cm": metrics.get("dir_cm", np.zeros((0, 0))).tolist()},
            )
            print(f"  saved {best_path}  (score={best_score:.4f})")

    # 保存最后一轮
    save_checkpoint(out_dir / "last.pt", model, epoch=args.epochs, metrics={"best_score": best_score})
    return best_path


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="CNN-LSTM 枪声/脚步声 + 方向训练")
    p.add_argument("--manifest", default=None)
    p.add_argument("--root", default=None, help="音频根目录（默认 manifest 所在目录）")
    p.add_argument("--synthetic", action="store_true", help="用合成数据跑通流程")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dir-weight", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--clip-grad", type=float, default=5.0)
    p.add_argument("--segment-seconds", type=float, default=1.0)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--fusion", choices=["attention", "stack"], default="attention")
    p.add_argument("--lstm-hidden", type=int, default=128)
    p.add_argument("--lstm-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-spatial-aux", action="store_true")
    p.add_argument("--no-spec-augment", action="store_true")
    p.add_argument("--mel-stats", default=None, help="compute_mel_stats.py 生成的 .npz")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="cpu（默认，AMD 推荐）/ cuda（NVIDIA）/ auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="CUDA 混合精度")
    p.add_argument("--max-batches", type=int, default=0, help="每轮最多多少步（调试）")
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="1 epoch + 合成数据 + 少量 batch")
    args = p.parse_args(argv)
    if args.smoke:
        args.synthetic = True
        args.epochs = 1
        args.batch_size = min(args.batch_size, 8)
        args.max_batches = args.max_batches or 3
        args.out = args.out if args.out != "runs/exp" else "runs/smoke"
    return args


if __name__ == "__main__":
    train(parse_args())
