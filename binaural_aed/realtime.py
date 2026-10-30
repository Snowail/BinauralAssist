"""实时双声道处理：Realtek 线路输入 → DSP → USB DAC。

典型用法::

    # 先看设备
    python -m binaural_aed.realtime --list

    # 延迟报告（依次尝试 64/128/256，固定 WASAPI 共享模式）
    python -m binaural_aed.realtime --latency-report --input-index 14 --output-index 12

    # 正式运行
    python -m binaural_aed.realtime --input-index 14 --output-index 12 ^
        --block-size 128 --monitor
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

import numpy as np
import sounddevice as sd

from .config import AudioConfig, PipelineConfig
from .dsp import db_to_lin, lin_to_db
from .enhance import BinauralPipeline


# ---------------------------------------------------------------------------
# 设备
# ---------------------------------------------------------------------------
def list_devices() -> None:
    hostapis = sd.query_hostapis()
    print(f"{'idx':>3}  {'in':>3} {'out':>3}  {'sr':>6}  hostapi        name")
    for i, dev in enumerate(sd.query_devices()):
        print(
            f"{i:>3}  {dev['max_input_channels']:>3} {dev['max_output_channels']:>3}  "
            f"{dev['default_samplerate']:>6.0f}  {hostapis[dev['hostapi']]['name']:<14} {dev['name']}"
        )


def resolve_device(query: str | None, index: int | None, kind: str, channels: int = 2) -> int:
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    if index is None and isinstance(query, int):
        index, query = query, None
    if index is not None:
        dev = sd.query_devices(index)
        if dev[key] < channels:
            raise RuntimeError(f"设备 [{index}] {dev['name']} 只有 {dev[key]} 个{kind}声道")
        return int(index)

    if query:
        matches = [
            (i, d)
            for i, d in enumerate(sd.query_devices())
            if d[key] >= channels and query.lower() in d["name"].lower()
        ]
        if not matches:
            raise RuntimeError(f"找不到名称包含 {query!r} 的 {kind} 设备（需要 {channels} 声道）")
        if len(matches) > 1:
            lines = "\n".join(f"  [{i}] {d['name']} ({sd.query_hostapis(d['hostapi'])['name']})" for i, d in matches)
            raise RuntimeError(f"匹配到多个设备，请用 --{kind}-index 指定：\n{lines}")
        return int(matches[0][0])

    default = sd.default.device[0 if kind == "input" else 1]
    if default is None or int(default) < 0:
        raise RuntimeError(f"没有默认 {kind} 设备，请指定 --{kind}-device / --{kind}-index")
    dev = sd.query_devices(int(default))
    if dev[key] < channels:
        raise RuntimeError(f"默认 {kind} 设备 {dev['name']} 只有 {dev[key]} 声道")
    return int(default)


def _wasapi_settings(device_index: int):
    try:
        hostapi = sd.query_hostapis(sd.query_devices(device_index)["hostapi"])
        if "WASAPI" in hostapi["name"].upper():
            return sd.WasapiSettings(exclusive=False)
    except Exception:
        pass
    return None


def open_stream(
    in_dev: int,
    out_dev: int,
    cfg: AudioConfig,
    callback,
) -> sd.Stream:
    """打开 WASAPI 共享模式双工流。"""
    extra = (_wasapi_settings(in_dev), _wasapi_settings(out_dev))
    if extra == (None, None):
        extra = None
    return sd.Stream(
        device=(in_dev, out_dev),
        channels=(cfg.channels, cfg.channels),
        samplerate=cfg.sample_rate,
        blocksize=cfg.block_size,
        dtype="float32",
        latency=cfg.latency,
        extra_settings=extra,
        callback=callback,
    )


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------
def run(cfg: AudioConfig, pipeline_cfg: PipelineConfig, seconds: float | None, monitor: bool) -> int:
    in_dev = resolve_device(cfg.input_device, None, "input", 2)
    out_dev = resolve_device(cfg.output_device, None, "output", 2)
    print(f"输入 [{in_dev}] {sd.query_devices(in_dev)['name']}")
    print(f"输出 [{out_dev}] {sd.query_devices(out_dev)['name']}")

    pipeline = BinauralPipeline(pipeline_cfg)
    stats = {"over": 0, "under": 0, "corr": 0.0, "rms": -120.0, "mono_warned": False}

    def callback(indata, outdata, frames, time_info, status):
        if status:
            if getattr(status, "input_overflow", False):
                stats["over"] += 1
            if getattr(status, "output_underflow", False):
                stats["under"] += 1
        try:
            left, right = indata[:, 0], indata[:, 1]
            rms = float(np.sqrt(np.mean(indata**2) + 1e-12))
            stats["rms"] = 20.0 * np.log10(max(rms, 1e-8))
            if np.std(left) > 1e-7 and np.std(right) > 1e-7:
                stats["corr"] = float(np.corrcoef(left, right)[0, 1])
                if stats["corr"] > 0.995 and stats["rms"] > -45 and not stats["mono_warned"]:
                    stats["mono_warned"] = True
                    print("\n警告：左右声道几乎完全相同，输入可能是单声道复制；方向增强会失效。")
            outdata[:] = pipeline.process(indata)
        except Exception:
            outdata.fill(0)

    stream = open_stream(in_dev, out_dev, cfg, callback)
    stream.start()
    lat_in, lat_out = stream.latency
    block_ms = 1000.0 * cfg.block_size / cfg.sample_rate
    print(
        f"延迟: input {lat_in*1000:.1f} ms + output {lat_out*1000:.1f} ms + block {block_ms:.1f} ms "
        f"≈ {(lat_in + lat_out)*1000 + block_ms:.1f} ms"
    )
    print("运行中（Ctrl+C 退出）...")

    t0 = time.perf_counter()
    try:
        while True:
            time.sleep(0.5)
            if seconds and time.perf_counter() - t0 >= seconds:
                break
            if not monitor:
                continue
            det = pipeline.last_detection
            if det is None:
                continue
            foot = float(det.foot_activity[-1])
            gun = float(det.gun_activity[-1])
            env = float(det.foot_env_db[-1])
            print(
                f"\rfoot {foot:4.2f}  gun {gun:4.2f}  foot_env {env:6.1f} dBFS  "
                f"in {stats['rms']:5.1f} dBFS  L/R corr {stats['corr']:+.2f}  "
                f"xruns {stats['over']}/{stats['under']}",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print()
    finally:
        stream.stop()
        stream.close()
    return 0


# ---------------------------------------------------------------------------
# 延迟报告
# ---------------------------------------------------------------------------
def latency_report(cfg: AudioConfig) -> int:
    in_dev = resolve_device(cfg.input_device, None, "input", 2)
    out_dev = resolve_device(cfg.output_device, None, "output", 2)
    print(f"输入 [{in_dev}] {sd.query_devices(in_dev)['name']}")
    print(f"输出 [{out_dev}] {sd.query_devices(out_dev)['name']}")
    print(f"采样率 {cfg.sample_rate} Hz\n")

    # 处理时间
    pipeline = BinauralPipeline(PipelineConfig(audio=cfg))
    block = np.zeros((cfg.block_size, 2), dtype=np.float32)
    for _ in range(20):
        pipeline.process(block)
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        pipeline.process(block)
        times.append((time.perf_counter() - t0) * 1000.0)
    print(f"DSP 处理: avg {np.mean(times):.3f} ms  p95 {np.percentile(times, 95):.3f} ms  "
          f"max {np.max(times):.3f} ms  (block={cfg.block_size})\n")

    print(f"{'block':>6} {'in_ms':>7} {'out_ms':>7} {'total_ms':>9}")
    for blocksize in (64, 128, 256):
        cfg2 = dataclasses.replace(cfg, block_size=blocksize)
        stats = {"frames": 0}

        def cb(indata, outdata, frames, time_info, status):
            outdata.fill(0)
            stats["frames"] += frames

        try:
            stream = open_stream(in_dev, out_dev, cfg2, cb)
            stream.start()
            time.sleep(0.2)
            lat_in, lat_out = stream.latency
            stream.stop()
            stream.close()
            total = (lat_in + lat_out) * 1000 + 1000 * blocksize / cfg.sample_rate + np.mean(times)
            print(
                f"{blocksize:>6} "
                f"{lat_in*1000:>7.1f} {lat_out*1000:>7.1f} {total:>9.1f}"
            )
        except Exception as exc:
            print(f"{blocksize:>6}  FAIL: {exc}")
    print("\n注意：以上是 PortAudio/WASAPI 报告的缓冲延迟，不含声卡 ADC/DAC 的模拟延迟（约 2–5 ms）和游戏自身缓冲。")
    return 0


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="双声道 HRTF 音频增强（脚步声 + 枪声平滑）")
    p.add_argument("--list", action="store_true")
    p.add_argument("--latency-report", action="store_true")
    p.add_argument("--input-device", default=None)
    p.add_argument("--input-index", type=int, default=None)
    p.add_argument("--output-device", default=None)
    p.add_argument("--output-index", type=int, default=None)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--monitor", action="store_true", help="打印检测活动")
    p.add_argument("--bypass", action="store_true", help="A/B 对比：直通不处理")
    p.add_argument("--input-gain-db", type=float, default=0.0)
    p.add_argument("--output-gain-db", type=float, default=0.0)
    p.add_argument("--max-boost-db", type=float, default=None)
    p.add_argument("--upward-threshold-db", type=float, default=None)
    p.add_argument("--foot-gate-db", type=float, default=None)
    p.add_argument("--foot-ratio", type=float, default=None)
    p.add_argument("--side-boost-db", type=float, default=None)
    p.add_argument("--foot-detail-db", type=float, default=None)
    p.add_argument("--gun-cut-db", type=float, default=None)
    p.add_argument("--limiter-db", type=float, default=None)
    args = p.parse_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        list_devices()
        return 0

    in_dev = resolve_device(args.input_device, args.input_index, "input", 2)
    out_dev = resolve_device(args.output_device, args.output_index, "output", 2)
    cfg = AudioConfig(
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        input_device=in_dev,
        output_device=out_dev,
        input_gain_db=args.input_gain_db,
        output_gain_db=args.output_gain_db,
    )
    pcfg = PipelineConfig(audio=cfg, bypass=args.bypass)
    if args.max_boost_db is not None:
        pcfg.footstep.max_boost_db = args.max_boost_db
    if args.upward_threshold_db is not None:
        pcfg.footstep.upward_threshold_db = args.upward_threshold_db
    if args.foot_gate_db is not None:
        pcfg.footstep.gate_db = args.foot_gate_db
    if args.foot_ratio is not None:
        pcfg.footstep.upward_ratio = args.foot_ratio
    if args.side_boost_db is not None:
        pcfg.footstep.side_boost_db = args.side_boost_db
    if args.foot_detail_db is not None:
        pcfg.footstep.detail_boost_db = args.foot_detail_db
    if args.gun_cut_db is not None:
        pcfg.gunshot.hf_cut_db = args.gun_cut_db
    if args.limiter_db is not None:
        pcfg.gunshot.limiter_threshold_db = args.limiter_db

    if args.latency_report:
        return latency_report(cfg)
    return run(pcfg.audio, pcfg, None if not args.seconds else args.seconds, args.monitor)


if __name__ == "__main__":
    raise SystemExit(main())
