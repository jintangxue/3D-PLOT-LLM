#!/usr/bin/env python
"""
Step 1: Extract per-shape, per-style point clouds from 3DCoMPaT200 HDF5,
keyed by 3DCoMPaT-GrIn ID convention.

3DCoMPaT200 valid_fine_2048_10.hdf5 layout:
  - 16920 entries = 1692 unique shapes × 10 styles (style 0..9)
  - HDF5 grouping: indices 0..1691 are style=0, 1692..3383 are style=1, etc.
  - Each entry: points (2048, 6) = xyz + rgb (u8 → /255 → float)

3DCoMPaT-GrIn ID format:  "<shape_id>_<style_idx>"  e.g. "05_0b1_0"
  - style_idx in 0..9 → use HDF5 entry at offset (style_idx*1692 + shape_pos)
  - shape_pos = position of shape_id in the (sorted within style block) list

Output:
  <out_dir>/{grin_id}_2048.npy   shape (2048, 6) float32, xyz + rgb in [0,1]

Usage:
  python 01_extract_pointclouds.py \\
      --hdf5 /tmp/compat_explore/.../valid_fine_2048_10.hdf5 \\
      --grin_jsons valid_single_part.json valid_multiple_parts.json valid_embodied.json \\
      --out_dir $COMPAT_DATA/npy
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def parse_grin_id(grin_id: str):
    """Parse a GrIn id into (shape_id, style_idx).

    Two formats observed in 3DCoMPaT-GrIn:
      - `valid_multiple_parts.json` / `valid_single_part.json`: `05_0b1_0`     (single underscore)
      - `valid_embodied.json`:                                 `23_056__0`   (double underscore — all style 0)
    Strip trailing underscores from shape_id so both map to the same HDF5 entry.
    """
    head, tail = grin_id.rsplit("_", 1)
    shape_id = head.rstrip("_")
    return shape_id, int(tail)


def build_index(hdf5_path: str):
    """Return {(shape_id, style_idx): hdf5_global_index}."""
    with h5py.File(hdf5_path, "r") as f:
        shape_ids = [s.decode() for s in f["shape_id"][:]]
        style_ids = [s.decode() for s in f["style_id"][:]]
    idx = {}
    for i, (s, st) in enumerate(zip(shape_ids, style_ids)):
        try:
            st_int = int(st)
        except ValueError:
            continue
        idx[(s, st_int)] = i
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, help="Path to valid_fine_2048_10.hdf5 (or train_)")
    ap.add_argument("--grin_jsons", nargs="+", required=True,
                    help="GrIn JSON files (each entry has 'id' field)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pointnum", type=int, default=2048,
                    help="Output point count. 2048 keeps native HDF5 size; pass 8192 to upsample.")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, process only first N ids (for testing).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect unique grin ids across all anno files
    all_ids = []
    seen = set()
    for jp in args.grin_jsons:
        with open(jp, "r") as f:
            rows = json.load(f)
        for r in rows:
            gid = r["id"]
            if gid not in seen:
                seen.add(gid)
                all_ids.append(gid)
    if args.limit > 0:
        all_ids = all_ids[: args.limit]
    print(f"[01] Total unique GrIn ids: {len(all_ids)}")

    # Parse and group by (shape_id, style_idx)
    parsed = []
    bad = 0
    for gid in all_ids:
        try:
            shape_id, style_idx = parse_grin_id(gid)
            parsed.append((gid, shape_id, style_idx))
        except (ValueError, IndexError):
            bad += 1
    if bad:
        print(f"[01] WARNING: {bad} unparseable ids skipped")

    # Build HDF5 index
    print(f"[01] Building HDF5 index from {args.hdf5}")
    idx_map = build_index(args.hdf5)
    print(f"[01] HDF5 indexed: {len(idx_map)} (shape, style) pairs")

    # Extract
    miss = 0
    skip_existing = 0
    written = 0
    with h5py.File(args.hdf5, "r") as f:
        points_ds = f["points"]
        for gid, shape_id, style_idx in tqdm(parsed, desc="extract"):
            out_path = out_dir / f"{gid}_{args.pointnum}.npy"
            if out_path.exists():
                skip_existing += 1
                continue
            key = (shape_id, style_idx)
            if key not in idx_map:
                miss += 1
                continue
            i = idx_map[key]
            pc = points_ds[i].astype(np.float32)  # (2048, 6) — xyz, rgb (rgb in 0..255 or 0..1?)

            # Normalize rgb to [0, 1] if it's in 0..255
            if pc[:, 3:].max() > 1.5:
                pc[:, 3:] = pc[:, 3:] / 255.0

            if args.pointnum != pc.shape[0]:
                # Upsample / downsample
                if args.pointnum > pc.shape[0]:
                    extra_idx = np.random.choice(pc.shape[0], args.pointnum - pc.shape[0], replace=True)
                    pc = np.concatenate([pc, pc[extra_idx]], axis=0)
                else:
                    keep_idx = np.random.choice(pc.shape[0], args.pointnum, replace=False)
                    pc = pc[keep_idx]

            np.save(out_path, pc)
            written += 1

    print(f"[01] Done. written={written}, skipped_existing={skip_existing}, missing_in_hdf5={miss}")


if __name__ == "__main__":
    main()
