"""实时方向增强：延迟线 + 方向增益 + 可选设备管线。

``DirectionalEnhancer`` 只做“采样块进、采样块出”，与声卡解耦，便于单测；
``run_device`` 演示如何用两条 sounddevice 流把它接到虚拟声卡和耳机上。

注意：本机目前没有 8 声道虚拟设备，``run_device`` 尚未在真实 8ch 链路上验证；
先用 ``--dry-run`` 验证算法，再按 README 装 Voicemeeter/Hi-Fi Cable。
"""

from __future__ import annotations

import argparse
import math
import queue
import threading
import time

import numpy as np
import torch

from .config import (
    CHANNEL_LAYOUTS,
    DIRECTION_CLASSES,
    EVENT_CLASSES,
    AudioConfig,
    direction_gain_db,
)


class EnhancerConfig:
    def __init__(
        self,
        delay_ms: float = 120.0,
        context_ms: float = 900.0,
        detect_interval_ms: float = 100.0,
        max_gain_db: float = 6.0,
        threshold: float = 0.45,
        attack_ms: float = 30.0,
        release_ms: float = 180.0,
        out_channels: int = 2,
    ):
        self.delay_ms = delay_ms
        self.context_ms = context_ms
        self.detect_interval_ms = detect_interval_ms
        self.max_gain_db = max_gain_db
        self.threshold = threshold
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.out_channels = out_channels


class DirectionalEnhancer:
    """输入 [frames, C] -> 增强后的 [frames, C]。

    处理顺序：
    1. 音频进入 delay 线（默认 120 ms），同时写入检测窗口；
    2. 每 100 ms 在窗口上跑一次模型；
    3. 枪声/脚步声置信度超过阈值时，按估计方向更新目标增益；
    4. 增益以 attack/release 平滑后乘到 **延迟后的** 音频块上。
    """

    def __init__(self, model, cfg: AudioConfig | None = None, enh: EnhancerConfig | None = None):
        self.model = model
        self.cfg = cfg or AudioConfig()
        self.enh = enh or EnhancerConfig()
        sr = self.cfg.sample_rate
        self.delay_samples = max(0, int(self.enh.delay_ms * sr / 1000.0))
        self.context_samples = int(self.enh.context_ms * sr / 1000.0)
        self.detect_samples = max(1, int(self.enh.detect_interval_ms * sr / 1000.0))

        self._window = np.zeros((self.context_samples, self.cfg.channels), dtype=np.float32)
        self._delay = np.zeros((0, self.cfg.channels), dtype=np.float32)
        self._since_detect = 0
        self.gains_db = np.zeros(self.cfg.channels, dtype=np.float32)
        self.target_db = np.zeros(self.cfg.channels, dtype=np.float32)
        self.last_event: str | None = None
        self.last_direction: str | None = None
        self.last_event_prob = 0.0
        self.detections = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._window[:] = 0
            self._delay = np.zeros((0, self.cfg.channels), dtype=np.float32)
            self._since_detect = 0
            self.gains_db[:] = 0
            self.target_db[:] = 0
            self.last_event = None
            self.last_direction = None
            self.detections = 0

    @torch.inference_mode()
    def _detect(self) -> None:
        audio = self._window.T[None, :, :].copy()  # [1, C, N]
        probs = self.model.predict(torch.from_numpy(audio))
        ev = probs["event_probs"][0]
        idx = int(ev.argmax())
        name = EVENT_CLASSES[idx]
        self.last_event = name
        self.last_event_prob = float(ev[idx])

        if name in ("gunshot", "footstep") and self.last_event_prob >= self.enh.threshold:
            dp = probs["direction_probs"][0]
            direction = DIRECTION_CLASSES[int(dp.argmax())]
            gains = direction_gain_db(direction, self.enh.max_gain_db)
            self.last_direction = direction
            self.target_db[:] = np.array(
                [gains[ch] for ch in CHANNEL_LAYOUTS[self.cfg.channels]], dtype=np.float32
            )
            self.detections += 1
        else:
            self.last_direction = None
            self.target_db[:] = 0.0

    def _smooth_gains(self, frames: int) -> None:
        sr = self.cfg.sample_rate
        attack = 1.0 - math.exp(-frames / max(1.0, self.enh.attack_ms * sr / 1000.0))
        release = 1.0 - math.exp(-frames / max(1.0, self.enh.release_ms * sr / 1000.0))
        delta = self.target_db - self.gains_db
        coef = np.where(delta > 0, attack, release).astype(np.float32)
        self.gains_db += coef * delta

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        frames = block.shape[0]

        with self._lock:
            self._delay = np.concatenate([self._delay, block], axis=0)
            if self._delay.shape[0] >= frames + self.delay_samples:
                out = self._delay[:frames].copy()
                self._delay = self._delay[frames:]
            else:
                out = np.zeros((frames, self.cfg.channels), dtype=np.float32)

            if frames >= self.context_samples:
                self._window[:] = block[-self.context_samples :]
            else:
                self._window = np.concatenate([self._window[frames:], block], axis=0)

            self._since_detect += frames
            if self._since_detect >= self.detect_samples:
                self._since_detect = 0
                self._detect()

            self._smooth_gains(frames)
            out *= (10.0 ** (self.gains_db / 20.0)).astype(np.float32)[None, :]
        return out

    def downmix(self, block: np.ndarray) -> np.ndarray:
        """8ch -> 2ch（7.1 常用下混系数）；out_channels=2 时使用。"""
        if self.enh.out_channels == block.shape[1]:
            return block
        if self.enh.out_channels != 2:
            raise ValueError("目前只实现了 8->2 下混")
        c = block.shape[1]
        if c == 2:
            return block
        left = block[:, 0].copy()
        right = block[:, 1].copy()
        if c >= 3:
            left += 0.707 * block[:, 2]
            right += 0.707 * block[:, 2]
        if c >= 6:
            left += 0.707 * block[:, 4]
            right += 0.707 * block[:, 5]
        if c >= 8:
            left += 0.5 * block[:, 6]
            right += 0.5 * block[:, 7]
        if c >= 4:
            left += 0.3 * block[:, 3]
            right += 0.3 * block[:, 3]
        return np.stack([left, right], axis=1).astype(np.float32)


def run_device(args) -> int:
    """把增强器接到真实声卡：8ch 虚拟设备 -> Python -> 2ch 耳机。"""
    import sounddevice as sd

    from .capture import _wasapi_settings, find_device
    from .checkpoint import load_model
    from .model import ModelConfig, MultiTaskAudioModel

    cfg = AudioConfig()
    in_dev = find_device(args.input_device, cfg.channels, "input", args.input_index)
    if args.output_index is not None:
        out_dev = find_device(None, args.out_channels, "output", args.output_index)
    elif args.output_device:
        out_dev = find_device(args.output_device, args.out_channels, "output", None)
    else:
        out_dev = int(sd.default.device[1])
    model = (
        load_model(args.checkpoint, "cpu")[0]
        if args.checkpoint
        else MultiTaskAudioModel(ModelConfig(audio=cfg)).eval()
    )

    enhancer = DirectionalEnhancer(
        model,
        cfg,
        EnhancerConfig(delay_ms=args.delay_ms, threshold=args.threshold, out_channels=args.out_channels),
    )
    in_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
    out_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
    stop = threading.Event()

    def in_callback(indata, frames, time_info, status):
        try:
            in_q.put_nowait(indata.copy())
        except queue.Full:
            pass

    def out_callback(outdata, frames, time_info, status):
        try:
            outdata[:] = out_q.get_nowait()
        except queue.Empty:
            outdata.fill(0)

    in_stream = sd.InputStream(
        device=in_dev,
        channels=cfg.channels,
        samplerate=cfg.sample_rate,
        blocksize=args.blocksize,
        dtype="float32",
        latency="low",
        callback=in_callback,
        extra_settings=_wasapi_settings(in_dev),
    )
    out_stream = sd.OutputStream(
        device=out_dev,
        channels=args.out_channels,
        samplerate=cfg.sample_rate,
        blocksize=args.blocksize,
        dtype="float32",
        latency="low",
        callback=out_callback,
        extra_settings=_wasapi_settings(out_dev),
    )

    def worker():
        while not stop.is_set():
            try:
                block = in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            enhanced = enhancer.process(block)
            stereo = enhancer.downmix(enhanced)
            try:
                out_q.put_nowait(stereo)
            except queue.Full:
                pass

    thread = threading.Thread(target=worker, daemon=True)
    in_stream.start()
    out_stream.start()
    thread.start()
    print(
        f"增强中：in=[{in_dev}] {cfg.channels}ch -> out=[{out_dev}] {args.out_channels}ch  "
        f"delay={args.delay_ms:.0f}ms  (Ctrl+C 退出)"
    )
    try:
        while True:
            time.sleep(0.5)
            print(
                f"\r{enhancer.last_event or '-':<10} p={enhancer.last_event_prob:4.2f}  "
                f"dir={enhancer.last_direction or '-':<20} det={enhancer.detections}",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print()
    finally:
        stop.set()
        in_stream.stop()
        in_stream.close()
        out_stream.stop()
        out_stream.close()
    return 0


def _dry_run(args) -> int:
    from .checkpoint import load_model
    from .dataset import SyntheticGameAudioDataset
    from .model import ModelConfig, MultiTaskAudioModel

    cfg = AudioConfig()
    model = (
        load_model(args.checkpoint, "cpu")[0]
        if args.checkpoint
        else MultiTaskAudioModel(ModelConfig(audio=cfg)).eval()
    )
    enhancer = DirectionalEnhancer(
        model, cfg, EnhancerConfig(delay_ms=args.delay_ms, threshold=args.threshold)
    )
    ds = SyntheticGameAudioDataset(1, cfg, train=False, seed=3)
    item = ds[0]
    audio = item["waveform"].T.numpy()
    true_ev = EVENT_CLASSES[int(item["event"])]

    outs = [enhancer.process(audio[i : i + 480]) for i in range(0, audio.shape[0], 480)]
    out = np.concatenate(outs, axis=0)
    stereo = enhancer.downmix(out)

    print(f"synthetic true_event={true_ev}  input={audio.shape}  output={out.shape}  stereo={stereo.shape}")
    print(f"delay={enhancer.delay_samples} samples  detections={enhancer.detections}")
    print(
        f"last_event={enhancer.last_event} p={enhancer.last_event_prob:.3f} "
        f"direction={enhancer.last_direction}"
    )
    print(f"gain_db={np.round(enhancer.gains_db, 2).tolist()}")
    if out.shape != audio.shape or stereo.shape[1] != 2:
        raise SystemExit("形状检查失败")
    print("dry-run OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="实时方向增强（延迟线 + 方向增益）")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--delay-ms", type=float, default=120.0)
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--blocksize", type=int, default=480)
    p.add_argument("--out-channels", type=int, default=2)
    p.add_argument("--input-device", default=None)
    p.add_argument("--input-index", type=int, default=None)
    p.add_argument("--output-device", default=None)
    p.add_argument("--output-index", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.dry_run:
        return _dry_run(args)
    if args.input_device or args.input_index:
        return run_device(args)
    p.error("指定 --dry-run，或用 --input-device/--input-index 接真实设备")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
