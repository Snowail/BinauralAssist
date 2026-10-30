"""离线自测：合成一段带左右/前后线索的“脚步 + 枪声”双声道音频，
跑完整 pipeline，输出前后 WAV 和客观指标。

运行::

    python -m binaural_aed.demo

输出目录 ``demo_out/``：
* before.wav / after.wav（可直接用耳机 A/B）
* 终端打印脚步提升、枪声峰值/高频衰减、ITD/ILD 变化、CPU 时间
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, sosfilt

from .config import AudioConfig, PipelineConfig
from .dsp import design_bandpass, design_highpass, lin_to_db
from .enhance import BinauralPipeline

SR = 48_000


def _env(n: int, decay_s: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.exp(-t / decay_s)


def _footstep(rng: np.random.Generator) -> np.ndarray:
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    env = _env(n, 0.022)
    thump = np.sin(2 * np.pi * 115 * t) * 0.7
    noise = rng.normal(0, 1, n) * 0.25
    sig = (thump + noise) * env
    hp = sosfilt(design_highpass(SR, 60), sig)
    return hp.astype(np.float32)


def _gunshot(rng: np.random.Generator) -> np.ndarray:
    n = int(0.22 * SR)
    env = _env(n, 0.028)
    noise = rng.normal(0, 1, n)
    sig = noise * env
    # 削弱低频，保留 2–8 kHz 的“刺耳”部分
    bp = sosfilt(design_bandpass(SR, 700, 10_000), sig)
    return bp.astype(np.float32)


def _add_stereo(
    out: np.ndarray,
    sig: np.ndarray,
    start_s: float,
    pan: float,
    itd_ms: float,
) -> None:
    """pan: -1 左 .. +1 右；itd_ms: 右耳相对左耳的时延（ms）。"""
    i0 = int(start_s * SR)
    n = min(len(sig), out.shape[0] - i0)
    if n <= 0:
        return
    sig = sig[:n]
    ild_db = pan * 6.0  # 正值 = 右声道更响
    g_left = 10 ** (-ild_db / 20)
    g_right = 10 ** (ild_db / 20)
    delay = int(round(abs(itd_ms) * SR / 1000.0))
    left = sig * g_left
    right = sig * g_right
    if delay > 0:
        if itd_ms >= 0:  # 右耳更晚：右声道延迟
            right = np.concatenate([np.zeros(delay, dtype=np.float32), right[:-delay]])
        else:
            left = np.concatenate([np.zeros(delay, dtype=np.float32), left[:-delay]])
    out[i0 : i0 + n, 0] += left
    out[i0 : i0 + n, 1] += right


def build_demo() -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(2.2 * SR)
    x = rng.normal(0, 10 ** (-58 / 20), (n, 2)).astype(np.float32)

    # 脚步：左、右、正前（正前 ITD/ILD 都接近 0）
    _add_stereo(x, _footstep(rng) * 0.035, 0.30, pan=-0.9, itd_ms=0.35)
    _add_stereo(x, _footstep(rng) * 0.035, 0.80, pan=+0.9, itd_ms=0.35)
    _add_stereo(x, _footstep(rng) * 0.030, 1.30, pan=0.0, itd_ms=0.0)

    # 枪声：偏右
    _add_stereo(x, _gunshot(rng) * 0.95, 1.70, pan=+0.7, itd_ms=0.25)

    peak = float(np.max(np.abs(x)))
    if peak > 0.98:
        x *= 0.98 / peak
    return x


def _band_db(x: np.ndarray, low: float, high: float) -> float:
    band = sosfilt(design_bandpass(SR, low, high), x, axis=0)
    return float(lin_to_db(np.sqrt(np.mean(band**2) + 1e-12)))


def _window(x: np.ndarray, center_s: float, width_s: float) -> np.ndarray:
    a = max(0, int((center_s - width_s / 2) * SR))
    b = min(x.shape[0], int((center_s + width_s / 2) * SR))
    return x[a:b]


def _itd_ild(x: np.ndarray) -> tuple[int, float]:
    left, right = x[:, 0], x[:, 1]
    if np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return 0, 0.0
    c = correlate(left, right, mode="full", method="fft")
    lags = np.arange(-len(left) + 1, len(right))
    max_lag = int(0.0015 * SR)
    mask = np.abs(lags) <= max_lag
    lag = int(lags[mask][np.argmax(c[mask])])
    ild = float(lin_to_db(np.sqrt(np.mean(left**2) + 1e-12)) - lin_to_db(np.sqrt(np.mean(right**2) + 1e-12)))
    return lag, ild


def main() -> int:
    cfg = PipelineConfig(audio=AudioConfig(sample_rate=SR, block_size=128))
    pipeline = BinauralPipeline(cfg)
    x = build_demo()

    blocks = []
    times = []
    for i in range(0, x.shape[0], cfg.audio.block_size):
        block = x[i : i + cfg.audio.block_size]
        t0 = time.perf_counter()
        blocks.append(pipeline.process(block))
        times.append((time.perf_counter() - t0) * 1000)
    y = np.concatenate(blocks, axis=0)

    # 指标
    foot_centers = (0.30, 0.80, 1.30)
    before_foot = np.mean([_band_db(_window(x, c, 0.14), 110, 2600) for c in foot_centers])
    after_foot = np.mean([_band_db(_window(y, c, 0.14), 110, 2600) for c in foot_centers])

    gun_before = _window(x, 1.70, 0.20)
    gun_after = _window(y, 1.70, 0.20)
    peak_before = float(np.max(np.abs(gun_before)))
    peak_after = float(np.max(np.abs(gun_after)))
    hf_before = _band_db(gun_before, 2500, 9000)
    hf_after = _band_db(gun_after, 2500, 9000)

    foot_win = _window(x, 0.30, 0.10)
    foot_win_after = _window(y, 0.30, 0.10)
    itd_b, ild_b = _itd_ild(foot_win)
    itd_a, ild_a = _itd_ild(foot_win_after)

    out_dir = Path("demo_out")
    out_dir.mkdir(exist_ok=True)
    sf.write(out_dir / "before.wav", x, SR, subtype="FLOAT")
    sf.write(out_dir / "after.wav", y, SR, subtype="FLOAT")

    print(f"脚步骤变（110–2600 Hz）: {after_foot - before_foot:+.2f} dB")
    print(f"枪声峰值: {lin_to_db(peak_after) - lin_to_db(peak_before):+.2f} dB")
    print(f"枪声 2.5–9 kHz 能量: {hf_after - hf_before:+.2f} dB")
    print(f"脚步 L/R ITD: {itd_b} -> {itd_a} samples")
    print(f"脚步 L/R ILD: {ild_b:+.2f} -> {ild_a:+.2f} dB")
    print(f"DSP: avg {np.mean(times):.3f} ms / block, max {np.max(times):.3f} ms, "
          f"实时系数 {np.mean(times) / (1000 * cfg.audio.block_size / SR):.3f}")
    print(f"写盘: {out_dir/'before.wav'} / {out_dir/'after.wav'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
