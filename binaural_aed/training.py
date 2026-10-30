"""训练素材库：保存实时截取的 2ch 片段、人工标签、导出 manifest。

数据布局::

    training_data/
      index.json              # 片段元数据（标签、时间、检测器分数）
      manifest.csv            # 导出给 fps_aed.train 的清单
      clips/
        20261019_170001_123_gunshot.wav
        ...

标签统一用英文 key，UI 显示中文；导出时映射到 fps_aed 的 event 类别。
"""

from __future__ import annotations

import csv
import json
import shutil
import threading
from datetime import datetime
from pathlib import Path

import soundfile as sf

LABELS: dict[str, dict] = {
    "gunshot": {"zh": "枪声", "event": "gunshot"},
    "footstep": {"zh": "脚步", "event": "footstep"},
    "other": {"zh": "其他", "event": "other"},
    "background": {"zh": "背景", "event": "background"},
    "unlabeled": {"zh": "待标注", "event": ""},
}


class TrainingStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.clips_dir = self.root / "clips"
        self.index_path = self.root / "index.json"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._items: list[dict] = []
        if self.index_path.exists():
            try:
                loaded = json.loads(self.index_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    self._items = loaded
            except Exception:
                self._items = []

    # -- persistence ----------------------------------------------------
    def _save(self) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._items, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.index_path)

    def _new_id(self) -> str:
        base = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        used = {item["id"] for item in self._items}
        candidate = base
        i = 1
        while candidate in used:
            candidate = f"{base}_{i}"
            i += 1
        return candidate

    # -- CRUD -----------------------------------------------------------
    def add(
        self,
        audio,
        sample_rate: int,
        label: str = "unlabeled",
        detector: dict | None = None,
    ) -> dict:
        if label not in LABELS:
            raise ValueError(f"未知标签: {label}")
        audio = audio.astype("float32", copy=False)
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("训练片段必须是 [N, 2] 立体声")
        if audio.shape[0] == 0:
            raise ValueError("空片段")

        with self._lock:
            self.clips_dir.mkdir(parents=True, exist_ok=True)
            item_id = self._new_id()
            filename = f"{item_id}_{label}.wav"
            path = self.clips_dir / filename
            sf.write(str(path), audio, sample_rate, subtype="PCM_24")
            item = {
                "id": item_id,
                "file": filename,
                "label": label,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "duration": round(audio.shape[0] / float(sample_rate), 3),
                "sample_rate": sample_rate,
                "channels": int(audio.shape[1]),
                "detector": detector or {},
            }
            self._items.append(item)
            self._save()
            return dict(item)

    def update_label(self, item_id: str, label: str) -> dict:
        if label not in LABELS:
            raise ValueError(f"未知标签: {label}")
        with self._lock:
            item = self._find(item_id)
            old_path = self.clips_dir / item["file"]
            new_name = f"{item['id']}_{label}.wav"
            new_path = self.clips_dir / new_name
            if old_path.exists() and old_path != new_path:
                old_path.rename(new_path)
            item["file"] = new_name
            item["label"] = label
            self._save()
            return dict(item)

    def delete(self, item_id: str) -> None:
        with self._lock:
            item = self._find(item_id)
            path = self.clips_dir / item["file"]
            if path.exists():
                path.unlink()
            self._items.remove(item)
            self._save()

    def _find(self, item_id: str) -> dict:
        for item in self._items:
            if item["id"] == item_id:
                return item
        raise KeyError(f"找不到片段: {item_id}")

    def audio_path(self, item_id: str) -> Path:
        with self._lock:
            item = self._find(item_id)
            path = (self.clips_dir / item["file"]).resolve()
            if self.clips_dir.resolve() not in path.parents:
                raise ValueError("非法路径")
            if not path.exists():
                raise FileNotFoundError(path)
            return path

    # -- views ----------------------------------------------------------
    def payload(self) -> dict:
        with self._lock:
            items = sorted(self._items, key=lambda x: x["created_at"], reverse=True)
            counts = {key: 0 for key in LABELS}
            for item in items:
                counts[item.get("label", "unlabeled")] = counts.get(item.get("label", "unlabeled"), 0) + 1
            return {
                "items": [dict(item) for item in items],
                "stats": counts,
                "labels": LABELS,
                "dir": str(self.root),
            }

    def export_manifest(self) -> dict:
        """导出给 fps_aed.train 的 manifest；未标注片段会被跳过。"""
        manifest = self.root / "manifest.csv"
        rows = 0
        counts: dict[str, int] = {}
        with self._lock, manifest.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "event", "direction", "split"])
            for item in sorted(self._items, key=lambda x: x["created_at"]):
                label = item.get("label", "unlabeled")
                event = LABELS.get(label, {}).get("event", "")
                if not event:
                    continue
                counts[event] = counts.get(event, 0) + 1
                split = "val" if rows > 0 and rows % 5 == 0 else "train"
                writer.writerow([f"clips/{item['file']}", event, "", split])
                rows += 1
        return {"path": str(manifest), "rows": rows}
