"""构建可解压即用的 Windows 便携包（BinauralAssist）。

依赖本机已安装的 Python 3.14 与项目 .venv（含 torch/scipy/flask 等）。
运行后生成 build/BinauralAssist，然后可用 tar 压缩。
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "build" / "BinauralAssist"
BASE_PYTHON = Path(os.environ.get("BINAURAL_BASE_PYTHON", sys.base_prefix))
VENV_SITE = Path(sysconfig.get_paths()["purelib"])


def copy_base_python() -> None:
    ignore = shutil.ignore_patterns(
        "site-packages", "__pycache__", "*.pyc", "*.pyo",
        "Doc", "include", "libs", "Scripts", "tcl",
        "test", "idlelib", "tkinter", "turtledemo", "ensurepip",
    )
    shutil.copytree(BASE_PYTHON, DEST / "python", ignore=ignore)


def copy_packages() -> None:
    target = DEST / "python" / "Lib" / "site-packages"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        VENV_SITE, target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        dirs_exist_ok=True,
    )


def copy_project() -> None:
    skip = {
        ".venv", ".git", "build", "build2", "build3", "build4", "build5", "runs", "training_data", "demo_out",
        ".uitest", ".tools", ".uvcache", ".pip-cache", ".ui_profile",
        ".edge_profile", ".edge_profile2", "waveforms", "__pycache__",
        "Binaural Assist.lnk", "lan_url.txt", "waves.png", ".gitignore", "packaging", "tray",
        "configs", "games", "tray_settings.json", "tray.log", "webui.log",
        "BinauralAssist-portable.zip.sha256", "goh",
    }
    for item in ROOT.iterdir():
        if item.name in skip or item.name.startswith(".tmp"):
            continue
        if item.name.endswith(".zip"):
            continue
        dest = DEST / item.name
        if item.is_dir():
            shutil.copytree(
                item, dest,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
        else:
            shutil.copy2(item, dest)


def main() -> None:
    if not BASE_PYTHON.exists():
        raise SystemExit(f"找不到基础 Python: {BASE_PYTHON}")
    if not VENV_SITE.exists():
        raise SystemExit(f"找不到项目依赖: {VENV_SITE}")
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    print("copy python runtime ...")
    copy_base_python()
    print("copy site-packages ...")
    copy_packages()
    print("copy project ...")
    copy_project()
    print("copy MSVC runtime ...")
    for dll in (
        "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll", "vcomp140.dll", "concrt140.dll",
    ):
        source = Path(r"C:\Windows\System32") / dll
        if source.exists():
            shutil.copy2(source, DEST / "python" / dll)

    # 便携说明
    portable_readme = """# Binaural Assist 便携版

## 解压即用

1. 解压整个 `BinauralAssist` 文件夹到任意位置（路径可以有空格/中文）；
2. 双击 **`Binaural Assist.exe`**（托盘图标使用 `trayicon.png`）；
3. 默认启动 WebUI；右下角托盘图标显示音频状态；双击图标打开控制台：
   - 本机：http://127.0.0.1:8765
   - 手机：http://<本机局域网 IP>:8765
4. **游戏识别** 默认包含三角洲行动；检测到游戏进程后自动开始音频处理，游戏全部退出后自动停止；
5. 右键托盘图标：音频状态、监听端口、局域网控制、游戏启动自动开启音频处理、开机启动、切换配置、关闭并退出。

`Binaural Assist.exe` 不会自动打开网页。WebUI 启动约 2–4 秒；第一次需要先在调音台手动启动一次，保存输入/输出设备。

## 说明

- 内置 Python 3.14 + torch/scipy/flask/sounddevice，无需安装 Python；
- 首次运行如果 Windows 防火墙弹窗，请允许“专用网络”；
- 手机与电脑需在同一局域网；托盘默认允许局域网，无密码；
- 参数会自动保存到 `configs/current.json`；UI 可保存/导入自定义配置；
- 训练素材在 `training_data/`，模型在 `runs/train2ch/`（首次使用会自动创建）；
- 完整使用说明见 `README_FULL.md`。
"""
    # 让根目录 README 直接是便携说明，完整开发文档保留为 README_FULL.md
    full = DEST / "README.md"
    if full.exists():
        full.replace(DEST / "README_FULL.md")
    (DEST / "README.md").write_text(portable_readme, encoding="utf-8")

    py = DEST / "python" / "python.exe"
    print(f"built: {DEST}")
    print(f"python: {py}")


if __name__ == "__main__":
    main()
