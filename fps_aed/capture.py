"""7.1 多声道采集：设备枚举、环形缓冲、WAV 录制。

注意
----
* 这里捕获的是“输入设备”。若游戏输出到 VB-Cable / Voicemeeter，请选择对应的
  录音端（例如 ``CABLE Output`` 或 ``Voicemeeter Out B1``）。
* Windows 的声道数由“声音控制面板 -> 播放设备 -> 配置”决定。只有把虚拟设备
  配置成 7.1（8 声道）并且游戏也输出 7.1，``channels=8`` 才能打开。
* VB-Cable 的 A/B 是 2 声道；8 声道请用 Voicemeeter Banana/Potato 或 Hi-Fi Cable。
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import CHANNEL_LAYOUTS, AudioConfig


def list_devices(kind: str | None = None, min_channels: int = 0) -> None:
    hostapis = sd.query_hostapis()
    print(f"{'idx':>3}  {'in':>3} {'out':>3}  {'sr':>6}  hostapi        name")
    for i, dev in enumerate(sd.query_devices()):
        if kind == "input" and dev["max_input_channels"] < max(1, min_channels):
            continue
        if kind == "output" and dev["max_output_channels"] < max(1, min_channels):
            continue
        host = hostapis[dev["hostapi"]]["name"]
        print(
            f"{i:>3}  {dev['max_input_channels']:>3} {dev['max_output_channels']:>3}  "
            f"{dev['default_samplerate']:>6.0f}  {host:<14} {dev['name']}"
        )


def find_device(name: str | None, channels: int, kind: str = "input", index: int | None = None) -> int:
    """按索引或名称子串选择设备，并检查声道数。"""
    if index is not None:
        dev = sd.query_devices(index)
        _check_channels(index, dev, channels, kind)
        return index

    candidates: list[tuple[int, dict]] = []
    for i, dev in enumerate(sd.query_devices()):
        key = "max_input_channels" if kind == "input" else "max_output_channels"
        if dev[key] < channels:
            continue
        if name is None or name.lower() in dev["name"].lower():
            candidates.append((i, dev))

    if not candidates:
        raise RuntimeError(
            f"找不到 {kind} 设备（至少 {channels} 声道，名称包含 {name!r}）。"
            "先用 --list 查看设备列表。"
        )
    if len(candidates) > 1:
        names = "\n".join(f"  [{i}] {d['name']}" for i, d in candidates)
        raise RuntimeError(f"匹配到多个设备，请用 --device-index 指定：\n{names}")
    index, dev = candidates[0]
    _check_channels(index, dev, channels, kind)
    return index


def _check_channels(index: int, dev: dict, channels: int, kind: str) -> None:
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    if dev[key] < channels:
        raise RuntimeError(
            f"设备 [{index}] {dev['name']} 只有 {dev[key]} 个{kind}声道，"
            f"需要 {channels}。请在 Windows 声音设置里把该设备配置为 7.1，"
            "或改用 8 声道的虚拟声卡（Voicemeeter / Hi-Fi Cable）。"
        )


def _wasapi_settings(device_index: int) -> sd.WasapiSettings | None:
    try:
        hostapi = sd.query_hostapis(sd.query_devices(device_index)["hostapi"])
        if "WASAPI" in hostapi["name"].upper():
            return sd.WasapiSettings(exclusive=False)
    except Exception:
        pass
    return None


class RingBuffer:
    """固定容量的多声道环形缓冲（写满后覆盖最旧数据）。"""

    def __init__(self, capacity: int, channels: int):
        self.capacity = capacity
        self.channels = channels
        self._buf = np.zeros((capacity, channels), dtype=np.float32)
        self._write = 0
        self._count = 0
        self._lock = threading.Lock()

    def push(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        n = block.shape[0]
        if n > self.capacity:
            block = block[-self.capacity :]
            n = self.capacity
        with self._lock:
            end = self._write + n
            if end <= self.capacity:
                self._buf[self._write : end] = block
            else:
                first = self.capacity - self._write
                self._buf[self._write :] = block[:first]
                self._buf[: end - self.capacity] = block[first:]
            self._write = end % self.capacity
            self._count = min(self.capacity, self._count + n)

    def latest(self, n: int) -> np.ndarray:
        n = min(n, self._count)
        if n <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)
        with self._lock:
            start = (self._write - n) % self.capacity
            if start + n <= self.capacity:
                return self._buf[start : start + n].copy()
            first = self.capacity - start
            return np.concatenate([self._buf[start:], self._buf[: n - first]], axis=0)

    def clear(self) -> None:
        with self._lock:
            self._write = 0
            self._count = 0


@dataclass
class CaptureConfig:
    audio: AudioConfig
    device: int | None = None
    blocksize: int = 480  # 10 ms @ 48 kHz
    buffer_seconds: float = 5.0
    latency: str = "low"


class MultichannelCapture:
    """后台采集，把音频写进环形缓冲；可注册回调做实时处理。"""

    def __init__(
        self,
        device: int | None = None,
        cfg: AudioConfig | None = None,
        blocksize: int = 480,
        buffer_seconds: float = 5.0,
        on_block=None,
    ):
        self.cfg = cfg or AudioConfig()
        self.device = device
        self.blocksize = blocksize
        self.ring = RingBuffer(
            int(buffer_seconds * self.cfg.sample_rate), self.cfg.channels
        )
        self.on_block = on_block
        self._stream: sd.InputStream | None = None
        self.underflows = 0
        self.overflows = 0

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            # input overflow 在调试时常见；记录但不打断采集
            if getattr(status, "input_overflow", False):
                self.overflows += 1
        self.ring.push(indata)
        if self.on_block is not None:
            self.on_block(indata.copy())

    def start(self) -> "MultichannelCapture":
        extra = _wasapi_settings(self.device) if self.device is not None else None
        self._stream = sd.InputStream(
            device=self.device,
            channels=self.cfg.channels,
            samplerate=self.cfg.sample_rate,
            blocksize=self.blocksize,
            dtype="float32",
            latency="low",
            callback=self._callback,
            extra_settings=extra,
        )
        self._stream.start()
        return self

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def read_latest(self, seconds: float) -> np.ndarray:
        return self.ring.latest(int(seconds * self.cfg.sample_rate))


def record_wav(
    path: str | Path,
    seconds: float | None,
    device: int | None = None,
    cfg: AudioConfig | None = None,
    subtype: str = "PCM_24",
    blocksize: int = 480,
) -> Path:
    """流式录制 8 声道 WAV；seconds=None 时录到 Ctrl+C。"""
    cfg = cfg or AudioConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
    extra = _wasapi_settings(device) if device is not None else None

    def callback(indata, frames, time_info, status):
        try:
            q.put_nowait(indata.copy())
        except queue.Full:
            pass

    stream = sd.InputStream(
        device=device,
        channels=cfg.channels,
        samplerate=cfg.sample_rate,
        blocksize=blocksize,
        dtype="float32",
        latency="low",
        callback=callback,
        extra_settings=extra,
    )

    levels = np.zeros(cfg.channels, dtype=np.float64)
    total = 0
    start = time.perf_counter()
    print(f"录制到 {path}  ({cfg.channels}ch / {cfg.sample_rate} Hz / {subtype})")
    if seconds:
        print(f"时长 {seconds:.1f}s ...")
    else:
        print("按 Ctrl+C 结束 ...")

    with sf.SoundFile(
        str(path), mode="w", samplerate=cfg.sample_rate, channels=cfg.channels, subtype=subtype
    ) as f:
        stream.start()
        try:
            while True:
                if seconds is not None and time.perf_counter() - start >= seconds:
                    break
                try:
                    block = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                f.write(block)
                levels += np.square(block, dtype=np.float64).sum(axis=0)
                total += block.shape[0]
        except KeyboardInterrupt:
            print("\n停止。")
        finally:
            stream.stop()
            stream.close()

    duration = total / cfg.sample_rate
    rms = np.sqrt(levels / max(total, 1))
    peak = 20 * np.log10(np.maximum(rms, 1e-8))
    print(f"完成：{duration:.2f}s，{total} samples")
    names = CHANNEL_LAYOUTS.get(cfg.channels, tuple(f"CH{i}" for i in range(cfg.channels)))
    for name, db in zip(names, peak):
        print(f"  {name:>3}: RMS {db:6.1f} dBFS")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="7.1 多声道采集 / 录制")
    parser.add_argument("--list", action="store_true", help="列出音频设备")
    parser.add_argument("--list-inputs", action="store_true", help="只列出输入设备")
    parser.add_argument("--device", default=None, help="设备名称子串，如 CABLE / Voicemeeter")
    parser.add_argument("--device-index", type=int, default=None, help="设备索引")
    parser.add_argument("--channels", type=int, default=8, choices=sorted(CHANNEL_LAYOUTS))
    parser.add_argument("--record", default=None, help="输出 WAV 路径")
    parser.add_argument("--seconds", type=float, default=10.0, help="录制时长；0 表示直到 Ctrl+C")
    parser.add_argument("--monitor", action="store_true", help="只打印各声道电平，不录制")
    args = parser.parse_args(argv)

    if args.list or args.list_inputs:
        list_devices("input" if args.list_inputs else None, args.channels)
        return 0

    cfg = AudioConfig(channels=args.channels)
    idx = find_device(args.device, args.channels, "input", args.device_index)
    dev = sd.query_devices(idx)
    print(f"使用输入设备 [{idx}] {dev['name']}  ({dev['max_input_channels']}ch)")

    if args.record:
        record_wav(args.record, None if args.seconds == 0 else args.seconds, idx, cfg)
        return 0

    # 电平监视
    cap = MultichannelCapture(idx, cfg).start()
    names = CHANNEL_LAYOUTS[args.channels]
    try:
        while True:
            time.sleep(0.25)
            block = cap.read_latest(0.1)
            if block.size == 0:
                continue
            rms = np.sqrt(np.square(block, dtype=np.float64).mean(axis=0))
            db = 20 * np.log10(np.maximum(rms, 1e-8))
            bars = " ".join(f"{n}:{v:6.1f}" for n, v in zip(names, db))
            print(f"\r{bars}", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        cap.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
