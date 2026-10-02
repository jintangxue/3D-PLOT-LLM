#!/usr/bin/env python
"""
Step 1b: Extract 8192-point clouds FROM RAW MESH (Compat200.zip) using the
official 3DCoMPaT-v2 StylizedShapeLoader.

Rationale: The pre-packaged HDF5 (`valid_fine_2048_10.hdf5`) only has 2048
points. Upsampling 2048 -> 8192 via random duplication (75% dup) degrades
feature quality. Sampling directly from the GLTF mesh gives genuine 8192
unique points, matching the methodology used by Kestrel / 3DCoMPaT paper.

Input:
  - zip_path:  Compat200.zip (24.8 GB, from HF CoMPaT/3DCoMPaT200)
  - meta_dir:  metadata/ (from github Vision-CAIR/3DCoMPaT-v2)
  - grin_jsons: GrIn anno JSON files (valid_*.json) — defines which
                (shape_id, style_id) pairs we actually need
  - split: 'valid' or 'train'

Output:
  - <out_dir>/{grin_id}_{pointnum}.npy  shape (pointnum, 6) float32, xyz+rgb in [0,1]

Usage:
  python 01b_extract_from_mesh.py \\
      --zip_path /path/to/Compat200.zip \\
      --meta_dir /path/to/3DCoMPaT-v2/metadata \\
      --grin_jsons valid_single_part.json valid_multiple_parts.json valid_embodied.json \\
      --out_dir $COMPAT_DATA/npy_mesh \\
      --split valid \\
      --pointnum 8192
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


# -- GrIn id parsing: tolerate both "05_0b1_0" and "23_056__0" formats
def parse_grin_id(grin_id: str):
    """Return (shape_id, style_id_str)."""
    head, tail = grin_id.rsplit("_", 1)
    shape_id = head.rstrip("_")
    return shape_id, tail  # keep style_id as str to match loader output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip_path", required=True)
    ap.add_argument("--meta_dir", required=True)
    ap.add_argument("--grin_jsons", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="valid", choices=["valid", "train", "test"])
    ap.add_argument("--semantic_level", default="fine", choices=["fine", "medium", "coarse"])
    ap.add_argument("--n_compositions", type=int, default=10,
                    help="Number of composition (style) indices to enumerate, matching HDF5 _10 convention")
    ap.add_argument("--pointnum", type=int, default=8192)
    ap.add_argument("--loader_path", default="",
                    help="Path to 3DCoMPaT-v2/loaders/3D directory (prepended to sys.path)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--worker_id", type=int, default=0,
                    help="Worker partition id (0..num_workers-1). Filters grin_ids by hash(gid) %% num_workers.")
    ap.add_argument("--num_workers", type=int, default=1,
                    help="Total number of parallel workers. >1 enables partitioning.")
    args = ap.parse_args()

    if args.loader_path:
        sys.path.insert(0, args.loader_path)
    from compat3D import StylizedShapeLoader  # noqa: E402

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build the set of (shape_id, style_id_str) pairs we actually need,
    # keyed to their GrIn IDs for output filename.
    # ------------------------------------------------------------------
    wanted = {}  # (shape_id, style_id_str) -> list of grin_ids that map here
    for jp in args.grin_jsons:
        with open(jp, "r") as f:
            rows = json.load(f)
        for r in rows:
            gid = r.get("id") or r.get("object_id")
            if gid is None:
                raise KeyError(f"row missing 'id'/'object_id': keys={list(r.keys())[:5]}")
            shape_id, style_id_str = parse_grin_id(gid)
            key = (shape_id, style_id_str)
            wanted.setdefault(key, []).append(gid)

    # Deduplicate grin_ids per key (sometimes same id appears in single+embodied)
    for k in wanted:
        wanted[k] = sorted(set(wanted[k]))

    print(f"[01b] unique (shape,style) pairs needed: {len(wanted)}")
    print(f"[01b] unique grin_ids to produce:        {sum(len(v) for v in wanted.values())}")

    # ------------------------------------------------------------------
    # Build the loader
    # ------------------------------------------------------------------
    print(f"[01b] Building StylizedShapeLoader ({args.split}/{args.semantic_level}, n_comp={args.n_compositions}, n_points={args.pointnum})")
    loader = StylizedShapeLoader(
        zip_path=args.zip_path,
        meta_dir=args.meta_dir,
        split=args.split,
        semantic_level=args.semantic_level,
        n_compositions=args.n_compositions,
        n_points=args.pointnum,
        get_mats=False,
    )
    n_total = len(loader)
    print(f"[01b] loader has {n_total} records")

    # Build loader index -> (shape_id, style_id_str) map
    # loader.model_style_ids[i] = [shape_id, style_id_str, comp_k]
    loader_index_by_key = {}
    for i, (sh, st, ck) in enumerate(loader.model_style_ids):
        key = (sh, st)
        # Multiple (comp_k) may share (sh, st); GrIn convention just uses first.
        # But more robustly, GrIn's style_id *is* the comp_k value. Let's check.
        loader_index_by_key.setdefault(key, []).append(i)

    # Diagnose mismatch: how many of our wanted keys exist in loader
    have = sum(1 for k in wanted if k in loader_index_by_key)
    miss_keys = [k for k in wanted if k not in loader_index_by_key]
    print(f"[01b] wanted keys found in loader: {have}/{len(wanted)}")
    if miss_keys[:5]:
        print(f"[01b] sample missing keys: {miss_keys[:5]}")

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------
    todo = []
    for key, grin_ids in wanted.items():
        if key not in loader_index_by_key:
            continue
        idx = loader_index_by_key[key][0]  # first composition with this (shape, style)
        for gid in grin_ids:
            # Worker partitioning: each gid hashes to one worker
            if args.num_workers > 1:
                # Use stable hash (not python's randomized one) — sum of byte values
                gid_hash = sum(gid.encode("utf-8"))
                if gid_hash % args.num_workers != args.worker_id:
                    continue
            out_path = out_dir / f"{gid}_{args.pointnum}.npy"
            if args.skip_existing and out_path.exists():
                continue
            todo.append((idx, gid, out_path))

    if args.limit > 0:
        todo = todo[: args.limit]

    print(f"[01b] will extract {len(todo)} npy files (skipped {sum(len(v) for v in wanted.values()) - len(todo)} existing/missing)")

    # Cache: multiple grin_ids may share the same loader index (same shape+style,
    # different GrIn annotation rows). Cache the loader[idx] result to avoid
    # re-sampling the same mesh.
    cache_idx = -1
    cache_pc = None
    cache_rgb = None

    t0 = time.time()
    written = 0
    failed = 0
    for loader_idx, gid, out_path in tqdm(todo, desc="extract-mesh"):
        try:
            if loader_idx != cache_idx:
                out = loader[loader_idx]
                # out = (shape_id, style_id, shape_label, pc, part_labels, colors)
                cache_idx = loader_idx
                cache_pc = out[3].astype(np.float32)   # (N, 3)
                cache_rgb = out[5].astype(np.float32)  # (N, 3), trimesh returns 0..255
                if cache_rgb.max() > 1.5:
                    cache_rgb = cache_rgb / 255.0      # normalize to [0, 1]
            pc6 = np.concatenate([cache_pc, cache_rgb], axis=1)  # (N, 6)
            if pc6.shape[0] != args.pointnum:
                # Shouldn't happen if loader n_points matches, but guard anyway
                raise ValueError(f"Unexpected n_points {pc6.shape[0]}")
            np.save(out_path, pc6)
            written += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [warn] {gid} (idx={loader_idx}) failed: {e}")

    dt = time.time() - t0
    print(f"[01b] DONE. written={written} failed={failed}  wall={dt/60:.1f} min  rate={written/dt:.2f}/s")


if __name__ == "__main__":
    main()
