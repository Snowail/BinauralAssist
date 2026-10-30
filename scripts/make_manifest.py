"""从按命名规范整理的录音目录生成 manifest.csv。

命名规范（双下划线分隔，方便解析带下划线的方向名）::

    gunshot__front_left_level__s01_0007.wav
    footstep__back_right_level__s01_0031.wav
    background__none__s01_0100.wav

目录约定：同一个父目录 = 同一次录制会话；脚本按会话整体划分 train/val，
避免同一段录音同时出现在训练和验证集里（数据泄漏）。

用法::

    python scripts/make_manifest.py --dir data/clips --out data/manifest.csv \
        --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fps_aed.config import DIRECTION_TO_IDX, EVENT_CLASSES  # noqa: E402


def parse_name(stem: str) -> tuple[str, str] | None:
    parts = stem.split("__")
    if len(parts) < 2:
        return None
    event = parts[0].strip()
    direction = parts[1].strip()
    if event not in EVENT_CLASSES:
        return None
    if event in ("gunshot", "footstep"):
        if direction not in DIRECTION_TO_IDX:
            return None
    else:
        direction = ""
    return event, direction


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--out", default="data/manifest.csv")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.dir)
    wavs = sorted(root.rglob("*.wav")) + sorted(root.rglob("*.flac"))
    if not wavs:
        raise SystemExit(f"{root} 下没有找到 wav/flac")

    # 以父目录作为“会话”
    sessions: dict[str, list[tuple[Path, str, str]]] = {}
    for wav in wavs:
        parsed = parse_name(wav.stem)
        if parsed is None:
            print(f"跳过（命名不符合规范）: {wav}")
            continue
        event, direction = parsed
        session = str(wav.parent.relative_to(root)) or "."
        sessions.setdefault(session, []).append((wav, event, direction))

    names = sorted(sessions)
    rng = random.Random(args.seed)
    rng.shuffle(names)
    n_val = int(round(len(names) * args.val_ratio))
    if args.val_ratio > 0 and len(names) > 1:
        n_val = max(1, min(len(names) - 1, n_val))
    if len(names) <= 1 and args.val_ratio > 0:
        print("警告：只有 1 个 session，无法按 session 划分验证集；请至少准备两个 session")
    val_sessions = set(names[:n_val])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "event", "direction", "split"])
        for session in sorted(sessions):
            split = "val" if session in val_sessions else "train"
            for wav, event, direction in sessions[session]:
                rel = os.path.relpath(wav.resolve(), out.parent.resolve())
                writer.writerow([Path(rel).as_posix(), event, direction, split])
                rows += 1

    print(f"wrote {out}: {rows} clips, {len(sessions)} sessions, {len(val_sessions)} val sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
