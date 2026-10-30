"""推理：离线 WAV 分析 + 实时监听（不含增强）。

示例::

    # 离线分析一段 8 声道录音
    python -m fps_aed.infer --wav data/session01.wav --checkpoint runs/baseline/best.pt

    # 实时监听虚拟声卡（只打印，不改音）
    python -m fps_aed.infer --realtime --device "Voicemeeter Out B1" \
        --checkpoint runs/baseline/best.pt

    # 没有模型/数据时用随机初始化模型检查链路
    python -m fps_aed.infer --synthetic
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model
from .config import (
    DIRECTION_CLASSES,
    EVENT_CLASSES,
    AudioConfig,
)
from .dataset import load_multichannel, SyntheticGameAudioDataset


def sliding_windows(audio: np.ndarray, window: int, hop: int) -> np.ndarray:
    """[N, C] -> [K, C, window]"""
    if audio.shape[0] < window:
        pad = window - audio.shape[0]
        audio = np.pad(audio, ((0, pad), (0, 0)))
    starts = list(range(0, audio.shape[0] - window + 1, hop))
    if not starts or starts[-1] + window < audio.shape[0]:
        starts.append(audio.shape[0] - window)
    return np.stack([audio[s : s + window].T for s in starts]).astype(np.float32)


@torch.inference_mode()
def analyze_audio(
    model,
    audio: np.ndarray,
    cfg: AudioConfig,
    window_seconds: float = 1.0,
    hop_seconds: float = 0.5,
    batch_size: int = 16,
):
    window = int(window_seconds * cfg.sample_rate)
    hop = int(hop_seconds * cfg.sample_rate)
    windows = sliding_windows(audio, window, hop)

    event_probs = []
    dir_probs = []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size])
        out = model.predict(batch)
        event_probs.append(out["event_probs"].cpu())
        dir_probs.append(out["direction_probs"].cpu())

    event_probs = torch.cat(event_probs)
    dir_probs = torch.cat(dir_probs)
    return {
        "event_probs": event_probs,       # [K, 4]
        "direction_probs": dir_probs,     # [K, 12]
        "event_mean": event_probs.mean(0),
        "direction_mean": dir_probs.mean(0),
        "n_windows": len(windows),
    }


def print_result(result: dict, topk: int = 3) -> None:
    ev = result["event_mean"]
    order = ev.argsort(descending=True)[:topk]
    print("事件概率（窗口平均）:")
    for i in order.tolist():
        print(f"  {EVENT_CLASSES[i]:<12} {float(ev[i])*100:6.2f}%")

    event_idx = int(ev.argmax())
    if EVENT_CLASSES[event_idx] in ("gunshot", "footstep"):
        dp = result["direction_mean"]
        order = dp.argsort(descending=True)[:topk]
        print("方向概率（窗口平均）:")
        for i in order.tolist():
            print(f"  {DIRECTION_CLASSES[i]:<20} {float(dp[i])*100:6.2f}%")
    else:
        print("当前窗口主要为背景/其他，方向分支不适用。")
    print(f"窗口数: {result['n_windows']}")


def run_realtime(args) -> int:
    from .capture import MultichannelCapture, find_device

    cfg = AudioConfig()
    device = find_device(args.device, cfg.channels, "input", args.device_index)
    model, _ = load_model(args.checkpoint, "cpu") if args.checkpoint else (None, None)
    if model is None:
        from .model import ModelConfig, MultiTaskAudioModel

        model = MultiTaskAudioModel(ModelConfig(audio=cfg)).eval()

    cap = MultichannelCapture(device, cfg, buffer_seconds=5.0).start()
    print("实时监听中（Ctrl+C 退出）...")
    try:
        while True:
            time.sleep(args.interval)
            audio = cap.read_latest(args.window)
            if audio.shape[0] < int(args.window * cfg.sample_rate):
                continue
            result = analyze_audio(model, audio, cfg, args.window, args.window, batch_size=1)
            ev = result["event_mean"]
            idx = int(ev.argmax())
            line = f"{EVENT_CLASSES[idx]:<10} {float(ev[idx])*100:5.1f}%"
            if EVENT_CLASSES[idx] in ("gunshot", "footstep"):
                dp = result["direction_mean"]
                di = int(dp.argmax())
                line += f"  |  {DIRECTION_CLASSES[di]:<20} {float(dp[di])*100:5.1f}%"
            print(f"\r{line}", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        cap.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FPS 音频事件/方向推理")
    p.add_argument("--wav", default=None, help="离线分析的 8 声道 WAV")
    p.add_argument("--checkpoint", default=None, help="best.pt；不填则随机初始化")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="用合成样本跑一遍")
    p.add_argument("--device", default=None, help="实时输入设备名称子串")
    p.add_argument("--device-index", type=int, default=None)
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--hop", type=float, default=0.5)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args(argv)

    cfg = AudioConfig(segment_seconds=args.window, channels=args.channels)
    device = "cpu"
    if args.checkpoint:
        model, _ = load_model(args.checkpoint, device)
        cfg = model.cfg.audio
    else:
        from .model import ModelConfig, MultiTaskAudioModel

        model = MultiTaskAudioModel(ModelConfig(audio=cfg)).to(device).eval()

    if args.realtime:
        return run_realtime(args)

    if args.synthetic:
        ds = SyntheticGameAudioDataset(4, cfg, train=False, seed=1)
        for i in range(len(ds)):
            item = ds[i]
            audio = item["waveform"].T.numpy()
            result = analyze_audio(model, audio, cfg, args.window, args.hop, batch_size=4)
            print(f"\n--- synthetic #{i}  true={EVENT_CLASSES[int(item['event'])]} "
                  f"/ {DIRECTION_CLASSES[int(item['direction'])] if int(item['direction'])>=0 else '-'}")
            print_result(result, args.topk)
        return 0

    if not args.wav:
        p.error("需要 --wav / --synthetic / --realtime 之一")

    audio = load_multichannel(args.wav, cfg.channels, cfg.sample_rate)
    print(f"{args.wav}: {audio.shape[0]/cfg.sample_rate:.2f}s, {audio.shape[1]}ch")
    result = analyze_audio(model, audio, cfg, args.window, args.hop)
    print_result(result, args.topk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
