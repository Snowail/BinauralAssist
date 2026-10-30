"""计算训练集对数梅尔谱的全局 mean/std，供 LogMelFrontend 归一化使用。

用法::

    python scripts/compute_mel_stats.py --manifest data/manifest.csv --samples 200 \
        --out runs/mel_stats.npz

    python scripts/compute_mel_stats.py --synthetic --samples 64 --out runs/mel_stats_synth.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fps_aed.config import AudioConfig  # noqa: E402
from fps_aed.dataset import GameAudioDataset, SyntheticGameAudioDataset  # noqa: E402
from fps_aed.features import LogMelSpectrogram  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default="runs/mel_stats.npz")
    args = p.parse_args()

    cfg = AudioConfig()
    if args.synthetic:
        ds = SyntheticGameAudioDataset(max(args.samples, 8), cfg, train=False, seed=0)
    else:
        if not args.manifest:
            p.error("需要 --manifest 或 --synthetic")
        ds = GameAudioDataset(args.manifest, args.root, cfg, train=True, augment=False)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    mel_fn = LogMelSpectrogram(cfg)
    n_mels = cfg.n_mels
    total = np.zeros(n_mels, dtype=np.float64)
    total_sq = np.zeros(n_mels, dtype=np.float64)
    count = 0

    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if i * args.batch_size >= args.samples:
                break
            mel = mel_fn(batch["waveform"])  # [B, C, M, T]
            arr = mel.numpy()
            total += arr.sum(axis=(0, 1, 3))
            total_sq += np.square(arr, dtype=np.float64).sum(axis=(0, 1, 3))
            count += arr.shape[0] * arr.shape[1] * arr.shape[3]

    mean = total / max(count, 1)
    var = total_sq / max(count, 1) - mean**2
    std = np.sqrt(np.maximum(var, 1e-8))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"saved {out}  frames={count}  mean={mean.mean():.3f}  std={std.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
