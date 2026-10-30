"""本地 Web UI：Flask 后端 + 实时音频引擎。

启动::

    python -m binaural_aed.webui --port 8765

浏览器打开 http://127.0.0.1:8765

设计上把“音频引擎”和“HTTP API”分开：
* AudioEngine 管理 sounddevice 流，在 PortAudio 回调线程里处理音频；
* Flask 只读取快照 / 修改参数，不参与实时音频路径；
* 运行中修改增益/阈值会立即生效；修改设备、block size 需要重启。
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import socket
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from . import __version__
from .buffer import RingBuffer
from .config import AudioConfig, PipelineConfig
from .enhance import BinauralPipeline
from .games import GameMonitor, GameStore, browse_exe
from .training import TrainingStore
from .training_runner import TrainingRunner

WEB_DIR = Path(__file__).resolve().parent / "web"
_LOG_STREAM = None


# ---------------------------------------------------------------------------
# 参数元数据：后端是唯一来源，前端据此渲染
# ---------------------------------------------------------------------------
PARAM_GROUPS = [
    {
        "id": "footstep",
        "title": "脚步增强",
        "subtitle": "Footstep enhancement",
        "accent": "cyan",
        "icon": "footsteps",
        "toggle": "footstep.enabled",
        "params": [
            {
                "key": "footstep.max_boost_db", "label": "最大提升", "min": 0, "max": 24,
                "step": 0.5, "unit": "dB", "desc": "只在检测到脚步瞬态时生效。调大能让很远的脚步更清楚，但会抬升环境底噪，也更容易伤听力；建议从 8–10 dB 开始。",
            },
            {
                "key": "footstep.upward_threshold_db", "label": "起始阈值", "min": -60, "max": -10,
                "step": 1, "unit": "dBFS", "desc": "低于该电平的声音才会被向上压缩。用 --monitor 的 foot_env 读数来定：安静脚步通常在 -50 ~ -30 dBFS。调高会让更多声音被抬。",
            },
            {
                "key": "footstep.upward_ratio", "label": "向上压缩比", "min": 1, "max": 6,
                "step": 0.1, "unit": ":1", "desc": "把阈值以下的声音往阈值方向抬多少。2.5 约等于把差距缩小到 40%；数值越大越激进，底噪也越明显。",
            },
            {
                "key": "footstep.gate_db", "label": "噪声门", "min": -90, "max": -30,
                "step": 1, "unit": "dBFS", "desc": "低于该电平不进行提升，避免放大风扇、电流声和键盘声。环境越安静可以设得越低。",
            },
            {
                "key": "footstep.side_boost_db", "label": "左右差异增强", "min": 0, "max": 12,
                "step": 0.5, "unit": "dB", "desc": "只放大 L-R 侧信号，增强左右定位（ILD），不改变 mid、不引入时延，因此 ITD 不变。过大时声像会显得不自然。",
            },
            {
                "key": "footstep.detail_boost_db", "label": "细节提亮", "min": -6, "max": 12,
                "step": 0.5, "unit": "dB", "desc": "短暂提亮 4–9 kHz 的鞋底摩擦/耳廓线索，帮助材质与前后判断；太大时齿音和底噪会明显。",
            },
            {
                "key": "footstep.attack_ms", "label": "起音时间", "min": 1, "max": 50,
                "step": 0.5, "unit": "ms", "desc": "增益包络的起音速度。太快会抽气，太慢会错过脚步起音；默认 4 ms。",
            },
            {
                "key": "footstep.release_ms", "label": "释放时间", "min": 20, "max": 500,
                "step": 5, "unit": "ms", "desc": "增益回落速度。太短会让脚步声“一闪一闪”，太长会把后续环境音一起抬高。",
            },
        ],
    },
    {
        "id": "gunshot",
        "title": "枪声平滑",
        "subtitle": "Gunshot smoothing",
        "accent": "amber",
        "icon": "gunshot",
        "toggle": "gunshot.enabled",
        "params": [
            {
                "key": "gunshot.hf_cut_db", "label": "高频衰减", "min": -18, "max": 0,
                "step": 0.5, "unit": "dB", "desc": "检测到枪声时衰减 2.5–9 kHz，降低刺耳感。切得越深越不刺激，但枪声会发闷，并削弱一部分前后/高度线索。",
            },
            {
                "key": "gunshot.limiter_threshold_db", "label": "限幅门限", "min": -24, "max": 0,
                "step": 0.5, "unit": "dBFS", "desc": "超过该峰值的枪声被联动限幅。越低越安全，但整体动态会被压扁；建议 -9 ~ -6 dBFS。",
            },
            {
                "key": "gunshot.limiter_release_ms", "label": "限幅释放", "min": 20, "max": 400,
                "step": 5, "unit": "ms", "desc": "限幅增益恢复速度。太短会产生“抽气”，太长会让连续枪声越来越小声。",
            },
            {
                "key": "gunshot.limiter_lookahead_ms", "label": "限幅预读", "min": 0, "max": 5,
                "step": 0.5, "unit": "ms", "desc": "提前读取峰值的时间。越大越能压住瞬间起音，但端到端延迟也线性增加；0 表示零延迟但可能漏掉最短的瞬态。",
            },
            {
                "key": "gunshot.detect_threshold_db", "label": "枪声检测门限", "min": -60, "max": -10,
                "step": 1, "unit": "dBFS", "desc": "2.5–9 kHz 高频能量超过该值才判定为枪声。太低会被脚步/环境误触发，太高会漏掉远距离枪声。",
            },
        ],
    },
    {
        "id": "output",
        "title": "输入 / 输出",
        "subtitle": "Gain & safety",
        "accent": "violet",
        "icon": "sliders",
        "toggle": None,
        "params": [
            {
                "key": "audio.input_gain_db", "label": "输入增益", "min": -24, "max": 24,
                "step": 0.5, "unit": "dB", "desc": "进入 DSP 前的数字增益。用来匹配线路电平；先保证最响枪声峰值约 -12 ~ -6 dBFS，再谈增强。",
            },
            {
                "key": "audio.output_gain_db", "label": "输出增益", "min": -24, "max": 24,
                "step": 0.5, "unit": "dB", "desc": "DSP 之后的增益。提高会让整体更响，但 ceiling 会软削波；建议保持 0，用系统音量控制响度。",
            },
            {
                "key": "output.ceiling_db", "label": "输出上限", "min": -12, "max": 0,
                "step": 0.5, "unit": "dBFS", "desc": "软削波上限，防止增强后爆音。建议 -1 dBFS 留一点余量。",
            },
        ],
    },
]

PRESETS = {
    "balanced": {
        "label": "平衡",
        "desc": "默认值：适度抬脚步、轻压枪声，适合大多数场景。",
        "patch": {},
    },
    "footsteps": {
        "label": "脚步优先",
        "desc": "更激进的向上压缩与侧向增强，适合安静环境/听声辨位训练。",
        "patch": {
            "footstep.max_boost_db": 18, "footstep.upward_threshold_db": -26,
            "footstep.upward_ratio": 3.0, "footstep.gate_db": -62,
            "footstep.side_boost_db": 5.0, "footstep.detail_boost_db": 3.0,
            "gunshot.hf_cut_db": -8,
        },
    },
    "competitive": {
        "label": "竞技",
        "desc": "较低延迟、保守增强，避免过度处理影响定位和听觉疲劳。",
        "patch": {
            "footstep.max_boost_db": 12, "footstep.upward_threshold_db": -34,
            "footstep.upward_ratio": 2.2, "footstep.side_boost_db": 3.0,
            "footstep.detail_boost_db": 1.0, "footstep.release_ms": 100,
            "gunshot.hf_cut_db": -6, "gunshot.limiter_threshold_db": -8,
            "gunshot.limiter_lookahead_ms": 1.0,
        },
    },
    "night": {
        "label": "夜间/护耳",
        "desc": "压低枪声、温和抬脚步，适合长时间使用。",
        "patch": {
            "footstep.max_boost_db": 8, "footstep.upward_threshold_db": -38,
            "footstep.gate_db": -55, "footstep.side_boost_db": 2.0,
            "footstep.detail_boost_db": 0.0,
            "gunshot.hf_cut_db": -12, "gunshot.limiter_threshold_db": -12,
        },
    },
}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _init_com() -> None:
    """Flask 请求运行在工作线程里，PortAudio/WASAPI 需要先初始化 COM。"""
    try:
        import ctypes
        ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
    except Exception:
        pass


def _wasapi_settings(device_index: int):
    try:
        hostapi = sd.query_hostapis(sd.query_devices(device_index)["hostapi"])
        if "WASAPI" in hostapi["name"].upper():
            return sd.WasapiSettings(exclusive=False)
    except Exception:
        pass
    return None


def _device_list(kind: str, min_channels: int = 2) -> list[dict]:
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    default = sd.default.device[0 if kind == "input" else 1]
    hostapis = sd.query_hostapis()
    out = []
    for i, dev in enumerate(sd.query_devices()):
        if dev[key] < min_channels:
            continue
        hostapi = hostapis[dev["hostapi"]]["name"]
        # 只展示真正可用的 WASAPI 端点；WDM-KS/MME/DirectSound 直接隐藏。
        if "WASAPI" not in hostapi.upper():
            continue
        out.append(
            {
                "index": i,
                "name": dev["name"],
                "channels": dev[key],
                "hostapi": hostapi,
                "default_samplerate": dev["default_samplerate"],
                "is_default": i == default,
            }
        )
    out.sort(key=lambda d: d["name"])
    return out


def _get(cfg: PipelineConfig, key: str):
    group, field = key.split(".", 1)
    if group == "audio":
        return getattr(cfg.audio, field)
    return getattr(getattr(cfg, group), field)


def _set(cfg: PipelineConfig, key: str, value):
    group, field = key.split(".", 1)
    if group == "audio":
        setattr(cfg.audio, field, value)
    else:
        setattr(getattr(cfg, group), field, value)


def _coerce(spec: dict, value):
    if isinstance(spec.get("min"), (int, float)):
        value = max(spec["min"], min(spec["max"], float(value)))
        if isinstance(spec.get("step"), int) and float(spec["step"]).is_integer():
            value = int(round(value))
        else:
            value = float(value)
    return value


def _param_snapshot(cfg: PipelineConfig) -> dict:
    values = {}
    for group in PARAM_GROUPS:
        for spec in group["params"]:
            values[spec["key"]] = _get(cfg, spec["key"])
    values["footstep.enabled"] = cfg.footstep.enabled
    values["gunshot.enabled"] = cfg.gunshot.enabled
    values["audio.block_size"] = cfg.audio.block_size
    values["audio.bypass"] = cfg.bypass
    return values


class ConfigStore:
    """配置预设：当前配置自动保存、自定义预设、导入/导出、上一次使用记录。"""

    def __init__(self, root: Path, builtins: dict):
        self.root = root
        self.builtins = builtins
        self.user_dir = root / "user"
        self.current_path = root / "current.json"
        self.state_path = root / "state.json"
        self.audio_path = root / "audio.json"
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read_json(self, path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_json(self, path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _safe_name(name: str) -> str:
        name = (name or "").strip()
        if not name or len(name) > 40 or any(c in name for c in '\\/:*?"<>|'):
            raise ValueError("预设名称无效（1–40 个字符，不能含路径字符）")
        return name

    def _reset_processing(self, cfg: PipelineConfig) -> None:
        defaults = PipelineConfig()
        for field in dataclasses.fields(defaults.footstep):
            setattr(cfg.footstep, field.name, getattr(defaults.footstep, field.name))
        for field in dataclasses.fields(defaults.gunshot):
            setattr(cfg.gunshot, field.name, getattr(defaults.gunshot, field.name))
        for field in dataclasses.fields(defaults.output):
            setattr(cfg.output, field.name, getattr(defaults.output, field.name))
        cfg.audio.input_gain_db = defaults.audio.input_gain_db
        cfg.audio.output_gain_db = defaults.audio.output_gain_db
        cfg.bypass = False

    def apply_params(self, cfg: PipelineConfig, params: dict, reset: bool = False) -> None:
        if reset:
            self._reset_processing(cfg)
        specs = {s["key"]: s for g in PARAM_GROUPS for s in g["params"]}
        for key, value in (params or {}).items():
            if key in ("footstep.enabled", "gunshot.enabled"):
                group = key.split(".", 1)[0]
                setattr(getattr(cfg, group), "enabled", bool(value))
            elif key == "audio.bypass":
                cfg.bypass = bool(value)
            elif key == "audio.block_size":
                block = int(value)
                if block in (64, 128, 256):
                    cfg.audio.block_size = block
            elif key in specs:
                _set(cfg, key, _coerce(specs[key], value))

    def current_params(self, cfg: PipelineConfig) -> dict:
        data = self._read_json(self.current_path)
        if isinstance(data, dict) and isinstance(data.get("params"), dict):
            return data["params"]
        return _param_snapshot(cfg)

    def save_current(self, cfg: PipelineConfig) -> dict:
        params = _param_snapshot(cfg)
        with self._lock:
            self._write_json(
                self.current_path,
                {"params": params, "saved_at": datetime.now().isoformat(timespec="seconds")},
            )
        return params

    def last_name(self) -> str:
        data = self._read_json(self.state_path)
        if isinstance(data, dict) and isinstance(data.get("last"), str):
            return data["last"]
        return "current"

    def set_last(self, name: str) -> None:
        with self._lock:
            self._write_json(self.state_path, {"last": name})

    def last_audio(self) -> dict:
        data = self._read_json(self.audio_path)
        return data if isinstance(data, dict) else {}

    def save_audio(self, cfg: PipelineConfig, device_names: dict) -> dict:
        data = {
            "input_device": cfg.audio.input_device,
            "output_device": cfg.audio.output_device,
            "input_name": (device_names or {}).get("input", ""),
            "output_name": (device_names or {}).get("output", ""),
            "block_size": cfg.audio.block_size,
        }
        with self._lock:
            self._write_json(self.audio_path, data)
        return data

    def items(self) -> list[dict]:
        items = [{"name": "current", "label": "当前配置", "desc": "自动保存的当前参数", "builtin": True}]
        for name, preset in self.builtins.items():
            items.append(
                {"name": name, "label": preset["label"], "desc": preset.get("desc", ""), "builtin": True}
            )
        for path in sorted(self.user_dir.glob("*.json")):
            data = self._read_json(path)
            if isinstance(data, dict):
                items.append(
                    {
                        "name": path.stem,
                        "label": data.get("label") or path.stem,
                        "desc": data.get("desc", "自定义配置"),
                        "builtin": False,
                    }
                )
        return items

    def payload(self, cfg: PipelineConfig) -> dict:
        return {
            "items": self.items(),
            "current": "current",
            "last": self.last_name(),
            "params": _param_snapshot(cfg),
        }

    def load(self, cfg: PipelineConfig, name: str) -> dict:
        name = (name or "current").strip()
        if name == "current":
            params = self.current_params(cfg)
            self.apply_params(cfg, params)
        elif name in self.builtins:
            self.apply_params(cfg, self.builtins[name].get("patch", {}), reset=True)
        else:
            path = self.user_dir / f"{self._safe_name(name)}.json"
            data = self._read_json(path)
            if not isinstance(data, dict):
                raise ValueError(f"找不到配置: {name}")
            self.apply_params(cfg, data.get("params", {}), reset=True)
        self.set_last(name)
        self.save_current(cfg)
        return {"ok": True, "name": name, "params": _param_snapshot(cfg)}

    def save_user(self, cfg: PipelineConfig, name: str, label: str | None = None) -> dict:
        name = self._safe_name(name)
        if name == "current" or name in self.builtins:
            raise ValueError("名称与内置配置冲突，请换一个名称")
        params = _param_snapshot(cfg)
        self._write_json(
            self.user_dir / f"{name}.json",
            {
                "label": label or name,
                "desc": "自定义配置",
                "params": params,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        self.set_last(name)
        self.save_current(cfg)
        return {"ok": True, "name": name, "params": params}

    def import_params(self, cfg: PipelineConfig, name: str, params: dict) -> dict:
        name = self._safe_name(name or "导入配置")
        if not isinstance(params, dict):
            raise ValueError("导入内容缺少 params 对象")
        self.apply_params(cfg, params, reset=True)
        self.save_user(cfg, name)
        return {"ok": True, "name": name, "params": _param_snapshot(cfg)}

    def delete(self, name: str) -> dict:
        name = self._safe_name(name)
        if name in self.builtins or name == "current":
            raise ValueError("内置配置不能删除")
        path = self.user_dir / f"{name}.json"
        if path.exists():
            path.unlink()
        if self.last_name() == name:
            self.set_last("current")
        return {"ok": True}


# ---------------------------------------------------------------------------
# 音频引擎
# ---------------------------------------------------------------------------
class AudioEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.cfg = PipelineConfig()
        self.pipeline: BinauralPipeline | None = None
        self.stream: sd.Stream | None = None
        self.running = False
        self.error: str | None = None
        self.needs_restart = False
        self.xruns = 0
        self.started_at: float | None = None
        self.latency = (0.0, 0.0)
        self.meters = {
            "rms_l_db": -120.0, "rms_r_db": -120.0,
            "peak": 0.0, "corr": 0.0,
            "foot": 0.0, "gun": 0.0, "foot_env_db": -120.0,
        }
        self.device_names = {"input": "", "output": ""}
        self.history: RingBuffer | None = None
        self.classifier = None
        self.classifier_path: str | None = None
        self.classifier_enabled = False
        self.config_store = None
        self.game_monitor = None

    # -- lifecycle ------------------------------------------------------
    def start(self, payload: dict) -> dict:
        with self.lock:
            self.stop()
            _init_com()
            cfg = self.cfg
            cfg.audio.input_device = int(payload.get("input_index", cfg.audio.input_device or 0))
            cfg.audio.output_device = int(payload.get("output_index", cfg.audio.output_device or 0))
            cfg.audio.block_size = int(payload.get("block_size", cfg.audio.block_size))
            if cfg.audio.block_size not in (64, 128, 256):
                raise ValueError("block_size 必须是 64 / 128 / 256")

            in_dev, out_dev = cfg.audio.input_device, cfg.audio.output_device
            for idx, kind, key in ((in_dev, "输入", "max_input_channels"), (out_dev, "输出", "max_output_channels")):
                info = sd.query_devices(idx)
                if info[key] < 2:
                    raise ValueError(f"{kind}设备 [{idx}] {info['name']} 不足 2 声道")
                hostapi = sd.query_hostapis(info["hostapi"])["name"]
                if "WASAPI" not in hostapi.upper():
                    raise ValueError(
                        f"{kind}设备 [{idx}] 使用的是 {hostapi}；本程序只支持 WASAPI 双工。"
                        "WDM-KS 会报 PaErrorCode -9993（illegal combination of I/O devices），"
                        "请改选 Windows WASAPI 端点。"
                    )
            self.device_names = {
                "input": sd.query_devices(in_dev)["name"],
                "output": sd.query_devices(out_dev)["name"],
            }

            self.pipeline = BinauralPipeline(cfg)
            self.history = RingBuffer(int(8 * cfg.audio.sample_rate), cfg.audio.channels)
            if self.classifier is not None and self.classifier_enabled:
                self.classifier.attach(self.history)
                self.classifier.start()
            extra = (_wasapi_settings(in_dev), _wasapi_settings(out_dev))
            if extra == (None, None):
                extra = None
            try:
                stream = sd.Stream(
                    device=(in_dev, out_dev),
                    channels=(2, 2),
                    samplerate=cfg.audio.sample_rate,
                    blocksize=cfg.audio.block_size,
                    dtype="float32",
                    latency="low",
                    extra_settings=extra,
                    callback=self._callback,
                )
                stream.start()
            except Exception as exc:
                raise RuntimeError(f"无法打开音频流：{exc}") from exc

            self.stream = stream
            self.running = True
            self.error = None
            self.needs_restart = False
            self.xruns = 0
            self.started_at = time.time()
            self.latency = stream.latency
            if self.config_store is not None:
                self.config_store.save_current(self.cfg)
                self.config_store.save_audio(self.cfg, self.device_names)
            return {"ok": True, "latency": self.latency}

    def _resolve_saved_device(self, kind: str, name: str | None, index, key: str) -> int:
        candidates = []
        for i, dev in enumerate(sd.query_devices()):
            if dev[key] < 2:
                continue
            hostapi = sd.query_hostapis(dev["hostapi"])["name"]
            if "WASAPI" not in hostapi.upper():
                continue
            candidates.append((i, dev["name"]))
        if name:
            exact = [i for i, n in candidates if n == name]
            if len(exact) == 1:
                return exact[0]
            partial = [i for i, n in candidates if name in n or n in name]
            if len(partial) == 1:
                return partial[0]
        if index is not None:
            try:
                idx = int(index)
                if any(i == idx for i, _ in candidates):
                    return idx
            except Exception:
                pass
        raise RuntimeError(f"找不到上次使用的{kind}设备：{name or index}")

    def start_from_last(self) -> dict:
        """按上次成功启动的设备名恢复音频处理（供托盘自动启动调用）。"""
        with self.lock:
            if self.running:
                return {"ok": True, "running": True, "already": True}
            if self.config_store is None:
                raise RuntimeError("配置存储未初始化")
            last = self.config_store.last_audio()
            if not last:
                raise RuntimeError("还没有保存过音频设备，请先在控制台手动启动一次")
            in_idx = self._resolve_saved_device("输入", last.get("input_name"), last.get("input_device"), "max_input_channels")
            out_idx = self._resolve_saved_device("输出", last.get("output_name"), last.get("output_device"), "max_output_channels")
            block = int(last.get("block_size", self.cfg.audio.block_size))
            result = self.start({
                "input_index": in_idx,
                "output_index": out_idx,
                "block_size": block,
            })
            return {**result, "running": True, "input_index": in_idx, "output_index": out_idx}

    def stop(self) -> None:
        with self.lock:
            if self.classifier is not None:
                self.classifier.stop()
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
            self.stream = None
            self.running = False
            self.pipeline = None
            self.history = None
            self.started_at = None
            self.error = None
            self.meters = {
                "rms_l_db": -120.0, "rms_r_db": -120.0,
                "peak": 0.0, "corr": 0.0,
                "foot": 0.0, "gun": 0.0, "foot_env_db": -120.0,
            }

    # -- audio callback -------------------------------------------------
    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.xruns += 1
        try:
            if self.history is not None:
                self.history.push(indata)
            if self.pipeline is not None:
                if self.classifier is not None and self.classifier_enabled:
                    foot, gun = self.classifier.probs()
                    self.pipeline.set_model_probs(foot, gun, True)
                else:
                    self.pipeline.set_model_probs(0.0, 0.0, False)
            y = self.pipeline.process(indata) if self.pipeline is not None else indata
        except Exception as exc:
            self.error = f"DSP 异常: {exc}"
            outdata.fill(0)
            return
        outdata[:] = y

        left, right = indata[:, 0], indata[:, 1]
        rms_l = float(np.sqrt(np.mean(left**2) + 1e-12))
        rms_r = float(np.sqrt(np.mean(right**2) + 1e-12))
        std_l, std_r = float(np.std(left)), float(np.std(right))
        corr = 0.0
        if std_l > 1e-7 and std_r > 1e-7:
            corr = float(np.mean((left - left.mean()) * (right - right.mean())) / (std_l * std_r))
        det = self.pipeline.last_detection if self.pipeline is not None else None
        self.meters = {
            "rms_l_db": 20.0 * np.log10(max(rms_l, 1e-8)),
            "rms_r_db": 20.0 * np.log10(max(rms_r, 1e-8)),
            "peak": float(np.max(np.abs(indata))),
            "corr": corr,
            "foot": float(det.foot_activity[-1]) if det is not None else 0.0,
            "gun": float(det.gun_activity[-1]) if det is not None else 0.0,
            "foot_env_db": float(det.foot_env_db[-1]) if det is not None else -120.0,
        }

    # -- runtime params -------------------------------------------------
    def update(self, patch: dict) -> dict:
        with self.lock:
            structural = {
                "audio.block_size", "audio.input_device",
                "audio.output_device", "gunshot.limiter_lookahead_ms",
            }
            for key, value in patch.items():
                if key in ("footstep.enabled", "gunshot.enabled"):
                    group = key.split(".", 1)[0]
                    setattr(getattr(self.cfg, group), "enabled", bool(value))
                    continue
                if key == "audio.bypass":
                    self.cfg.bypass = bool(value)
                    if self.pipeline is not None:
                        self.pipeline.bypass = bool(value)
                    continue
                if key in structural:
                    _set(self.cfg, key, value)
                    self.needs_restart = True
                    continue
                spec = next(
                    (s for g in PARAM_GROUPS for s in g["params"] if s["key"] == key), None
                )
                if spec is None:
                    raise ValueError(f"未知参数: {key}")
                _set(self.cfg, key, _coerce(spec, value))
            if self.pipeline is not None:
                self.pipeline.refresh_gains()
            if self.config_store is not None:
                self.config_store.save_current(self.cfg)
                self.config_store.set_last("current")
            return {"ok": True, "needs_restart": self.needs_restart}

    def apply_preset(self, name: str) -> dict:
        if self.config_store is not None:
            old_block = self.cfg.audio.block_size
            result = self.config_store.load(self.cfg, name)
            if self.pipeline is not None:
                self.pipeline.refresh_gains()
            self.needs_restart = old_block != self.cfg.audio.block_size
            return {**result, "needs_restart": self.needs_restart}
        preset = PRESETS.get(name)
        if preset is None:
            raise ValueError(f"未知预设: {name}")
        self.update(preset["patch"])
        return {"ok": True, "params": _param_snapshot(self.cfg)}

    def reset(self) -> dict:
        with self.lock:
            defaults = PipelineConfig()
            # 保留设备/block，只把处理参数恢复默认；注意要原地修改，
            # 因为 pipeline 里的 detector/enhancer 持有这些 dataclass 的引用。
            for field in dataclasses.fields(defaults.footstep):
                setattr(self.cfg.footstep, field.name, getattr(defaults.footstep, field.name))
            for field in dataclasses.fields(defaults.gunshot):
                setattr(self.cfg.gunshot, field.name, getattr(defaults.gunshot, field.name))
            for field in dataclasses.fields(defaults.output):
                setattr(self.cfg.output, field.name, getattr(defaults.output, field.name))
            self.cfg.audio.input_gain_db = defaults.audio.input_gain_db
            self.cfg.audio.output_gain_db = defaults.audio.output_gain_db
            self.cfg.bypass = False
            if self.pipeline is not None:
                self.pipeline.refresh_gains()
            if self.config_store is not None:
                self.config_store.save_current(self.cfg)
                self.config_store.set_last("current")
            return {"ok": True, "params": _param_snapshot(self.cfg)}

    # -- training capture ------------------------------------------------
    def capture_recent(self, seconds: float) -> np.ndarray:
        if not self.running or self.history is None:
            raise RuntimeError("音频未运行，先点“启动音频”")
        n = int(round(seconds * self.cfg.audio.sample_rate))
        data = self.history.latest(n)
        if data.shape[0] < n:
            raise RuntimeError(f"缓冲不足（需要 {n/self.cfg.audio.sample_rate:.1f}s），请先运行一会儿")
        return data

    def load_classifier(self, checkpoint: str, enabled: bool = True) -> dict:
        """加载 2ch 事件模型；运行中会立即接管检测器。"""
        from .classifier import ModelEventClassifier

        if self.classifier is not None:
            self.classifier.stop()
        self.classifier = ModelEventClassifier(
            checkpoint,
            sample_rate=self.cfg.audio.sample_rate,
            window_seconds=1.0,
            interval_seconds=0.2,
        )
        self.classifier_path = str(checkpoint)
        self.classifier_enabled = bool(enabled)
        if self.running and self.history is not None:
            self.classifier.attach(self.history)
            self.classifier.start()
        return self.classifier.status()

    def set_classifier_enabled(self, enabled: bool) -> dict:
        self.classifier_enabled = bool(enabled)
        if self.classifier is not None:
            if enabled and self.running and self.history is not None:
                self.classifier.attach(self.history)
                self.classifier.start()
            elif not enabled:
                self.classifier.stop()
        return self.classifier_status()

    def classifier_status(self) -> dict:
        if self.classifier is None:
            return {"loaded": False, "enabled": self.classifier_enabled, "checkpoint": self.classifier_path}
        data = self.classifier.status()
        data["enabled"] = self.classifier_enabled
        return data

    def detector_snapshot(self) -> dict:
        det = self.pipeline.last_detection if self.pipeline is not None else None
        if det is None:
            return {"foot": 0.0, "gun": 0.0, "foot_env_db": -120.0}
        return {
            "foot": float(np.max(det.foot_activity)),
            "gun": float(np.max(det.gun_activity)),
            "foot_env_db": float(np.mean(det.foot_env_db)),
        }

    # -- status ---------------------------------------------------------
    def status(self) -> dict:
        lat_in, lat_out = self.latency if self.running else (0.0, 0.0)
        block_ms = 1000.0 * self.cfg.audio.block_size / self.cfg.audio.sample_rate
        return {
            "running": self.running,
            "error": self.error,
            "needs_restart": self.needs_restart,
            "latency": {
                "input_ms": lat_in * 1000.0,
                "output_ms": lat_out * 1000.0,
                "block_ms": block_ms,
                "total_ms": (lat_in + lat_out) * 1000.0 + block_ms,
            },
            "xruns": self.xruns,
            "meters": self.meters,
            "devices": self.device_names,
            "block_size": self.cfg.audio.block_size,
            "bypass": self.cfg.bypass,
            "classifier": self.classifier_status(),
            "uptime": (time.time() - self.started_at) if self.started_at else 0.0,
        }

    def bootstrap(self) -> dict:
        return {
            "version": __version__,
            "groups": PARAM_GROUPS,
            "presets": PRESETS,
            "configs": self.config_store.payload(self.cfg) if self.config_store is not None else None,
            "last_audio": self.config_store.last_audio() if self.config_store is not None else {},
            "params": _param_snapshot(self.cfg),
            "devices": {
                "inputs": _device_list("input"),
                "outputs": _device_list("output"),
            },
            "status": self.status(),
            "capabilities": _capabilities(),
        }


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
def _capabilities() -> dict:
    try:
        import torch
        return {
            "torch": torch.__version__,
            "cuda": bool(torch.cuda.is_available()),
            "cpu_count": os.cpu_count() or 1,
        }
    except Exception as exc:
        return {"cuda": False, "error": str(exc)}


def _lan_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return None


def create_app(engine: AudioEngine | None = None, password: str | None = None) -> Flask:
    engine = engine or AudioEngine()
    root = Path(__file__).resolve().parent.parent
    store = TrainingStore(root / "training_data")
    runner = TrainingRunner(root, store, on_success=engine.load_classifier)
    configs = ConfigStore(root / "configs", PRESETS)
    try:
        configs.load(engine.cfg, configs.last_name())
    except Exception:
        configs.load(engine.cfg, "current")
    engine.config_store = configs

    game_env = os.environ.get("BINAURAL_GAME_AUTO")
    game_override = None if game_env is None else game_env.strip().lower() not in ("0", "false", "no", "off")
    games = GameStore(root / "games", auto_override=game_override)
    monitor = GameMonitor(games, engine)
    monitor.start()
    engine.game_monitor = monitor

    app = Flask(__name__, static_folder=str(WEB_DIR / "static"), static_url_path="/static")
    app.json.ensure_ascii = False

    if password:
        @app.before_request
        def _auth():
            auth = request.authorization
            if not auth or auth.password != password:
                return Response(
                    "需要密码", 401,
                    {"WWW-Authenticate": 'Basic realm="Binaural Assist"'},
                )
            return None

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/api/bootstrap")
    def bootstrap():
        return jsonify(engine.bootstrap())

    @app.get("/api/status")
    def status():
        return jsonify(engine.status())

    @app.get("/api/devices")
    def devices():
        return jsonify({"inputs": _device_list("input"), "outputs": _device_list("output")})

    @app.post("/api/start")
    def start():
        data = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(engine.start(data))
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/autostart")
    def autostart():
        try:
            return jsonify(engine.start_from_last())
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/stop")
    def stop():
        engine.stop()
        return jsonify({"ok": True})

    @app.post("/api/shutdown")
    def shutdown():
        """停止音频并彻底结束 Flask 进程（退出按钮使用）。"""
        engine.stop()

        def _exit():
            time.sleep(0.6)
            os._exit(0)

        threading.Thread(target=_exit, daemon=True).start()
        return jsonify({"ok": True})

    @app.post("/api/params")
    def params():
        data = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(engine.update(data.get("patch", {})))
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/preset")
    def preset():
        data = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(engine.apply_preset(data.get("name", "")))
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/reset")
    def reset():
        return jsonify(engine.reset())

    # -- configuration presets -----------------------------------------
    @app.get("/api/configs")
    def configs_get():
        return jsonify(configs.payload(engine.cfg))

    @app.post("/api/configs/save")
    def configs_save():
        data = request.get_json(force=True, silent=True) or {}
        try:
            saved = configs.save_user(engine.cfg, str(data.get("name", "")), data.get("label"))
            return jsonify({"ok": True, **saved, **configs.payload(engine.cfg)})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/configs/load")
    def configs_load():
        data = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(engine.apply_preset(str(data.get("name", "current"))))
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/configs/import")
    def configs_import():
        data = request.get_json(force=True, silent=True) or {}
        try:
            result = configs.import_params(
                engine.cfg,
                str(data.get("name", "导入配置")),
                data.get("params", {}),
            )
            if engine.pipeline is not None:
                engine.pipeline.refresh_gains()
            return jsonify({"ok": True, **result, **configs.payload(engine.cfg)})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/configs/delete")
    def configs_delete():
        data = request.get_json(force=True, silent=True) or {}
        try:
            configs.delete(str(data.get("name", "")))
            return jsonify({"ok": True, **configs.payload(engine.cfg)})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    # -- game process whitelist ----------------------------------------
    def games_payload() -> dict:
        status = monitor.status()
        return games.payload(set(status.get("detected", [])), status)

    @app.get("/api/games")
    def games_list():
        return jsonify(games_payload())

    @app.post("/api/games/browse")
    def games_browse():
        try:
            path = browse_exe()
            if not path:
                return jsonify({"ok": False, "cancelled": True})
            return jsonify({"ok": True, "path": path})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/games/add")
    def games_add():
        data = request.get_json(force=True, silent=True) or {}
        try:
            games.add(str(data.get("path", "")), data.get("display_name"))
            return jsonify({"ok": True, **games_payload()})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/games/update")
    def games_update():
        data = request.get_json(force=True, silent=True) or {}
        try:
            games.update(
                str(data.get("id", "")),
                data.get("display_name"),
                data.get("enabled"),
            )
            return jsonify({"ok": True, **games_payload()})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/games/delete")
    def games_delete():
        data = request.get_json(force=True, silent=True) or {}
        try:
            games.delete(str(data.get("id", "")))
            return jsonify({"ok": True, **games_payload()})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/games/auto")
    def games_auto():
        data = request.get_json(force=True, silent=True) or {}
        try:
            monitor.set_enabled(bool(data.get("enabled", True)))
            return jsonify({"ok": True, **games_payload()})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/api/games/icon/<item_id>")
    def game_icon(item_id: str):
        path = games.icon_path(item_id)
        if path is None:
            return jsonify({"ok": False, "error": "没有图标"}), 404
        return send_file(path, max_age=0)

    @app.post("/api/games/icon/<item_id>")
    def game_icon_upload(item_id: str):
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return jsonify({"ok": False, "error": "没有选择图标"}), 400
        try:
            games.set_icon(item_id, upload.read(), Path(upload.filename).suffix)
            return jsonify({"ok": True, **games_payload()})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    # -- training data -------------------------------------------------
    @app.get("/api/train")
    def train_list():
        return jsonify(store.payload())

    @app.post("/api/train/capture")
    def train_capture():
        data = request.get_json(force=True, silent=True) or {}
        label = data.get("label", "unlabeled")
        try:
            seconds = max(0.2, min(5.0, float(data.get("seconds", 1.0))))
            audio = engine.capture_recent(seconds)
            det = engine.detector_snapshot()
            item = store.add(audio, engine.cfg.audio.sample_rate, label, det)
            suggested = "unlabeled"
            if det.get("gun", 0) >= 0.35 and det.get("gun", 0) >= det.get("foot", 0) * 0.8:
                suggested = "gunshot"
            elif det.get("foot", 0) >= 0.35:
                suggested = "footstep"
            elif det.get("foot", 0) < 0.05 and det.get("gun", 0) < 0.05:
                suggested = "background"
            payload = store.payload()
            return jsonify({"ok": True, "item": item, "suggested": suggested, "stats": payload["stats"]})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/train/label")
    def train_label():
        data = request.get_json(force=True, silent=True) or {}
        try:
            item = store.update_label(str(data.get("id", "")), str(data.get("label", "")))
            return jsonify({"ok": True, "item": item, "stats": store.payload()["stats"]})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/train/delete")
    def train_delete():
        data = request.get_json(force=True, silent=True) or {}
        try:
            store.delete(str(data.get("id", "")))
            return jsonify({"ok": True, "stats": store.payload()["stats"]})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/api/train/audio/<item_id>")
    def train_audio(item_id: str):
        try:
            return send_file(store.audio_path(item_id), mimetype="audio/wav", conditional=True)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404

    @app.post("/api/train/export")
    def train_export():
        try:
            return jsonify({"ok": True, **store.export_manifest()})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/api/train/job")
    def train_job():
        return jsonify(runner.snapshot())

    @app.post("/api/train/start")
    def train_start():
        data = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify({"ok": True, **runner.start(
                device=data.get("device", "cpu"),
                epochs=data.get("epochs", 20),
                batch_size=data.get("batch_size", 16),
                auto_replace=data.get("auto_replace", True),
            )})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/train/stop")
    def train_stop():
        return jsonify({"ok": True, **runner.stop()})

    @app.get("/api/classifier")
    def classifier_get():
        return jsonify(engine.classifier_status())

    @app.post("/api/classifier")
    def classifier_set():
        data = request.get_json(force=True, silent=True) or {}
        try:
            enabled = bool(data.get("enabled", False))
            checkpoint = data.get("checkpoint")
            if checkpoint or (enabled and engine.classifier is None):
                path = Path(checkpoint) if checkpoint else root / "runs" / "train2ch" / "best.pt"
                if not path.exists():
                    raise RuntimeError(f"找不到模型：{path}；先训练一次")
                return jsonify({"ok": True, **engine.load_classifier(str(path), enabled)})
            return jsonify({"ok": True, **engine.set_classifier_enabled(enabled)})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "not found"}), 404

    return app


def _ensure_output_streams() -> None:
    """pythonw.exe 下 sys.stdout/stderr 为 None，Flask 日志会直接崩溃。"""
    global _LOG_STREAM
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_path = Path(__file__).resolve().parent.parent / "webui.log"
    try:
        stream = open(log_path, "a", encoding="utf-8", buffering=1)
    except Exception:
        stream = open(os.devnull, "w", encoding="utf-8")
    _LOG_STREAM = stream
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def main(argv: list[str] | None = None) -> int:
    _ensure_output_streams()
    p = argparse.ArgumentParser(description="Binaural Assist Web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    p.add_argument("--password", default=None, help="局域网访问密码（HTTP Basic）")
    args = p.parse_args(argv)

    app = create_app(password=args.password)
    local_url = f"http://127.0.0.1:{args.port}"
    print(f"Binaural Assist UI: {local_url}")
    if args.host in ("0.0.0.0", "::"):
        ip = _lan_ip()
        if ip:
            print(f"LAN:                 http://{ip}:{args.port}")
        if not args.password:
            print("警告：当前监听 0.0.0.0 且没有密码，局域网内任何人都能控制。建议加 --password。")
    if args.open:
        threading.Timer(1.0, lambda: webbrowser.open(local_url)).start()
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
