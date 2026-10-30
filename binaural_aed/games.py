"""游戏进程白名单、图标、以及“检测到游戏后自动开始音频处理”的监控。"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import threading
import time
import uuid
from ctypes import wintypes
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .webui import AudioEngine

DEFAULT_GAME_PATH = Path(
    r"C:\WeGameApps\rail_apps\DeltaForce(2001918)\DeltaForce\Binaries\Win64"
    r"\DeltaForceClient-Win64-Shipping.exe"
)

_CREATE_NO_WINDOW = 0x08000000
_ICON_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _powershell(script: str, env: dict | None = None, timeout: int = 20, capture: bool = False):
    flags = _CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-STA", "-Command", script],
        env=env,
        capture_output=capture,
        text=True,
        timeout=timeout,
        creationflags=flags,
    )


def extract_icon(exe_path: Path, out_path: Path) -> bool:
    """用 Windows Shell 的关联图标接口把 exe 图标存成 PNG。"""
    if os.name != "nt" or not exe_path.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if out_path.exists():
            out_path.unlink()
    except Exception:
        pass
    env = os.environ.copy()
    env["BA_EXE_PATH"] = str(exe_path)
    env["BA_ICON_PATH"] = str(out_path)
    script = (
        "Add-Type -AssemblyName System.Drawing; "
        "$i=[System.Drawing.Icon]::ExtractAssociatedIcon($env:BA_EXE_PATH); "
        "if ($i -eq $null) { exit 2 }; "
        "$b=$i.ToBitmap(); "
        "$b.Save($env:BA_ICON_PATH,[System.Drawing.Imaging.ImageFormat]::Png); "
        "exit 0"
    )
    try:
        result = _powershell(script, env=env, timeout=20)
        return result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def browse_exe() -> str | None:
    """在服务端桌面弹出 Windows 文件选择框，返回 exe 完整路径。"""
    if os.name != "nt":
        return None
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$d=New-Object System.Windows.Forms.OpenFileDialog; "
        "$d.Title='选择游戏 EXE'; "
        "$d.Filter='可执行程序 (*.exe)|*.exe|所有文件 (*.*)|*.*'; "
        "$d.CheckFileExists=$true; "
        "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
        "[Console]::Out.Write($d.FileName) }"
    )
    try:
        result = _powershell(script, timeout=300, capture=True)
        return (result.stdout or "").strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 进程枚举：Toolhelp32，取完整 exe 名称（tasklist 可能截断长名称）
# ---------------------------------------------------------------------------
_TH32CS_SNAPPROCESS = 0x00000002
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


def process_names() -> set[str] | None:
    """返回当前所有进程的 exe 文件名（小写）；读取失败返回 None。"""
    if os.name != "nt":
        return set()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    handle = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if handle == _INVALID_HANDLE_VALUE:
        return None
    names: set[str] = set()
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        if not kernel32.Process32FirstW(handle, ctypes.byref(entry)):
            return None
        while True:
            name = (entry.szExeFile or "").strip()
            if name:
                names.add(name.lower())
            if not kernel32.Process32NextW(handle, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(handle)
    return names


class GameStore:
    def __init__(self, root: Path, auto_override: bool | None = None):
        self.root = root
        self.icons_dir = root / "icons"
        self.path = root / "games.json"
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.icons_dir.mkdir(parents=True, exist_ok=True)
        first_run = not self.path.exists()
        self.data = self._load()
        if auto_override is not None:
            self.data["auto_enabled"] = bool(auto_override)
        if first_run:
            # 首次运行：加入默认的三角洲行动条目，并尝试提取 exe 自带图标。
            self._create_default()
        elif auto_override is not None:
            self._save()

    def _load(self) -> dict:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("items"), list):
                raw.setdefault("auto_enabled", True)
                return raw
        except Exception:
            pass
        return {"version": 1, "auto_enabled": True, "items": []}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _create_default(self) -> None:
        item = {
            "id": "deltaforce",
            "path": str(DEFAULT_GAME_PATH),
            "exe_name": DEFAULT_GAME_PATH.name,
            "display_name": "三角洲行动",
            "enabled": True,
            "icon": "",
        }
        icon_name = "deltaforce.png"
        if extract_icon(DEFAULT_GAME_PATH, self.icons_dir / icon_name):
            item["icon"] = icon_name
        self.data["items"].append(item)
        self._save()

    def _public_item(self, item: dict, detected_lower: set[str]) -> dict:
        return {
            "id": item.get("id", ""),
            "path": item.get("path", ""),
            "exe_name": item.get("exe_name", ""),
            "display_name": item.get("display_name") or item.get("exe_name", ""),
            "enabled": bool(item.get("enabled", True)),
            "icon": bool(item.get("icon")),
            "detected": (item.get("exe_name", "").lower() in detected_lower),
        }

    def items(self) -> list[dict]:
        with self.lock:
            return [dict(x) for x in self.data.get("items", [])]

    def has_item(self, item_id: str) -> bool:
        return any(x.get("id") == item_id for x in self.data.get("items", []))

    def add(self, path: str, display_name: str | None = None) -> dict:
        p = Path((path or "").strip().strip('"'))
        if not p.exists() or not p.is_file():
            raise ValueError("文件不存在")
        if p.suffix.lower() != ".exe":
            raise ValueError("请选择 .exe 可执行程序")
        exe_name = p.name
        with self.lock:
            if any(x.get("exe_name", "").lower() == exe_name.lower() for x in self.data.get("items", [])):
                raise ValueError("这个进程已经在列表里了")
            item_id = "game_" + uuid.uuid4().hex[:8]
            icon_name = ""
            icon_file = self.icons_dir / f"{item_id}.png"
            if extract_icon(p, icon_file):
                icon_name = icon_file.name
            item = {
                "id": item_id,
                "path": str(p),
                "exe_name": exe_name,
                "display_name": (display_name or p.stem).strip()[:60],
                "enabled": True,
                "icon": icon_name,
            }
            self.data["items"].append(item)
            self._save()
            return dict(item)

    def update(self, item_id: str, display_name: str | None = None, enabled: bool | None = None) -> dict:
        with self.lock:
            for item in self.data.get("items", []):
                if item.get("id") != item_id:
                    continue
                if display_name is not None:
                    name = display_name.strip()
                    if not name:
                        raise ValueError("显示名称不能为空")
                    item["display_name"] = name[:60]
                if enabled is not None:
                    item["enabled"] = bool(enabled)
                self._save()
                return dict(item)
        raise ValueError("找不到这个游戏")

    def delete(self, item_id: str) -> None:
        with self.lock:
            keep = []
            removed = None
            for item in self.data.get("items", []):
                if item.get("id") == item_id:
                    removed = item
                else:
                    keep.append(item)
            if removed is None:
                raise ValueError("找不到这个游戏")
            self.data["items"] = keep
            self._save()
            icon = removed.get("icon")
            if icon:
                try:
                    (self.icons_dir / icon).unlink()
                except Exception:
                    pass

    def set_icon(self, item_id: str, data: bytes, suffix: str) -> str:
        suffix = (suffix or "").lower()
        if suffix not in _ICON_SUFFIXES:
            raise ValueError("图标只支持 PNG/JPG/WebP/GIF/BMP")
        if not data or len(data) > 4 * 1024 * 1024:
            raise ValueError("图标文件为空或超过 4 MB")
        with self.lock:
            for item in self.data.get("items", []):
                if item.get("id") != item_id:
                    continue
                filename = f"{item_id}{suffix}"
                (self.icons_dir / filename).write_bytes(data)
                old = item.get("icon")
                item["icon"] = filename
                if old and old != filename:
                    try:
                        (self.icons_dir / old).unlink()
                    except Exception:
                        pass
                self._save()
                return filename
        raise ValueError("找不到这个游戏")

    def icon_path(self, item_id: str) -> Path | None:
        with self.lock:
            for item in self.data.get("items", []):
                if item.get("id") == item_id and item.get("icon"):
                    path = self.icons_dir / item["icon"]
                    if path.exists():
                        return path
        return None

    @property
    def auto_enabled(self) -> bool:
        return bool(self.data.get("auto_enabled", True))

    def set_auto(self, enabled: bool) -> None:
        with self.lock:
            self.data["auto_enabled"] = bool(enabled)
            self._save()

    def payload(self, detected_lower: set[str], monitor: dict) -> dict:
        with self.lock:
            items = [self._public_item(x, detected_lower) for x in self.data.get("items", [])]
        return {
            "items": items,
            "auto_enabled": self.auto_enabled,
            "detected_count": sum(1 for x in items if x["detected"]),
            **monitor,
        }


class GameMonitor(threading.Thread):
    def __init__(self, store: GameStore, engine: "AudioEngine", interval: float = 1.0):
        super().__init__(name="game-monitor", daemon=True)
        self.store = store
        self.engine = engine
        self.interval = interval
        self._stop = threading.Event()
        self.lock = threading.RLock()
        self.detected: set[str] = set()
        self.was_running = False
        self.auto_started = False
        self.last_error = ""
        self.last_check = 0.0

    def stop(self) -> None:
        self._stop.set()

    def set_enabled(self, enabled: bool) -> None:
        self.store.set_auto(enabled)
        with self.lock:
            self.was_running = False
            self.last_error = ""

    def status(self) -> dict:
        with self.lock:
            return {
                "auto_enabled": self.store.auto_enabled,
                "detected": sorted(self.detected),
                "auto_started": self.auto_started,
                "last_error": self.last_error,
                "audio_running": bool(self.engine.running),
            }

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                names = process_names()
                if names is None:
                    with self.lock:
                        self.last_error = "进程列表读取失败"
                    self._stop.wait(self.interval)
                    continue
                items = self.store.items()
                whitelist = {str(item.get("exe_name", "")).lower() for item in items if item.get("exe_name")}
                with self.lock:
                    self.detected = names & whitelist
                    self.last_check = time.time()
                    present = any(
                        item.get("enabled", True)
                        and item.get("exe_name", "").lower() in names
                        for item in items
                    )
                    auto = self.store.auto_enabled

                    if auto and present and not self.was_running and not self.engine.running:
                        try:
                            self.engine.start_from_last()
                            self.auto_started = True
                            self.last_error = ""
                        except Exception as exc:
                            self.last_error = str(exc)
                            self.auto_started = False

                    # 游戏全部退出：同步停止音频处理（只在自动模式下生效；
                    # 手动模式完全由用户控制，不跟随游戏进程）。
                    if auto and self.was_running and not present and self.engine.running:
                        try:
                            self.engine.stop()
                            self.auto_started = False
                            self.last_error = ""
                        except Exception as exc:
                            self.last_error = f"停止音频失败: {exc}"

                    self.was_running = present if auto else False
            except Exception as exc:
                with self.lock:
                    self.last_error = f"进程检测失败: {exc}"
            self._stop.wait(self.interval)
