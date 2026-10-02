"""
PartVerse canonical layout (mesh + annotations).

The point-cloud NPZ under `cache_global_rgb_nn_partid/` is derived from:
  - normalized_glbs / textured_part_glbs / anno_infos (see the PartVerse release).

This module only resolves paths and loads lightweight JSON; no trimesh dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PartVersePaths:
    root: Path
    anno_infos: Path
    text_captions_json: Path
    normalized_glbs: Path
    textured_part_glbs: Path
    point_cache_dir: Path  # *.npz with global_xyz, point_part_id, ...

    @classmethod
    def from_root(cls, root: str | Path, point_cache_dir: str | Path) -> PartVersePaths:
        r = Path(root)
        return cls(
            root=r,
            anno_infos=r / "anno_infos",
            text_captions_json=r / "text_captions.json",
            normalized_glbs=r / "normalized_glbs",
            textured_part_glbs=r / "textured_part_glbs",
            point_cache_dir=Path(point_cache_dir),
        )


def load_caption_index(text_captions_json: Path) -> Dict[str, Any]:
    with open(text_captions_json, "r") as f:
        return json.load(f)


def get_object_captions(caption_index: Dict[str, Any], obj_id: str) -> Optional[Dict[str, List[str]]]:
    """
    Returns view-keyed captions. Values are lists of strings (short + long, etc.).
    Keys are stringified view indices as stored in PartVerse (e.g. '0','1',...).
    """
    entry = caption_index.get(obj_id)
    if entry is None:
        return None
    return entry


def load_object_info(anno_infos: Path, obj_id: str) -> Optional[Dict[str, Any]]:
    p = anno_infos / obj_id / f"{obj_id}_info.json"
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


def list_localized_part_glbs(textured_root: Path, obj_id: str) -> List[Path]:
    d = textured_root / obj_id
    if not d.is_dir():
        return []
    return sorted(d.glob("*.glb"))
