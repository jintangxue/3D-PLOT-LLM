#!/usr/bin/env python3
"""
PartVerse semantic parts <-> BCP groups (K-way) for Stage-2 supervision.

Data sources:
  - PartVerse mesh annotations live under partverse/anno_infos, captions in text_captions.json.
  - Point NPZ (global_xyz, point_part_id, ...) is your mesh-sampled cache — not the raw glb.
  - Objaverse 8192 npy + pc_norm matches PointLLM / BCP pack frame (patch_xyz in pack).

Alignment: ICP + Umeyama (unordered 8192 vs 8192 FPS).

Point → patch → BCP group (default: PointBERT-consistent):
  Each Objaverse point is assigned a primary patch among the 512 FPS centers using the same
  inverse rule as PointLLM PointBERT ``Group`` (``group_size`` nearest points per patch; a point
  claimed by multiple patches picks the patch with smallest center distance; if none, Voronoi).
  Then ``grp_per_point = patch_rK[patch_idx]``. Legacy Voronoi-only lift:
  ``--point_patch_assign voronoi``.

Regenerate after changing assign mode (new JSONL → new stage2 part JSON → re-run build_splits.py):
  python -m data_tools.partverse_qa.bcp_semantic_mapping build \\
    --out outputs/mappings_k16_pointbert.jsonl
  python -m data_tools.partverse_qa.build_stage2_part_json \\
    --mappings outputs/mappings_k16_pointbert.jsonl \\
    --out outputs/stage2_part_k16_iou04_parttok_pointbert.json \\
    --min_union_iou 0.4
  # Then filter/merge/eval: filter_stage2_part_json.py + build_splits.py with updated paths.

Usage:
  python -m data_tools.partverse_qa.bcp_semantic_mapping verify --obj_id <uuid>
  python -m data_tools.partverse_qa.bcp_semantic_mapping build --out outputs/mappings_k16.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree as _CKDTree
except ImportError:
    _CKDTree = None

# Repo defaults (override with CLI)
try:
    from .default_paths import (
        BCP_PACK_DIR,
        OBJAVERSE_8192_DIR,
        OUTPUT_DIR,
        PARTVERSE_POINT_CACHE,
    )
except ImportError:
    from default_paths import (  # type: ignore
        BCP_PACK_DIR,
        OBJAVERSE_8192_DIR,
        OUTPUT_DIR,
        PARTVERSE_POINT_CACHE,
    )


def nn_argmin(queries: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Nearest ref index per query row. Prefers cKDTree (O(N log M)) if scipy is installed."""
    if _CKDTree is not None:
        _, idx = _CKDTree(refs).query(queries, k=1)
        return np.asarray(idx, dtype=np.int32)
    d2 = np.sum((queries[:, None, :] - refs[None, :, :]) ** 2, axis=-1)
    return np.argmin(d2, axis=1).astype(np.int32)


def pc_norm_numpy(pc: np.ndarray) -> np.ndarray:
    xyz = pc[:, :3].astype(np.float64)
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    xyz = xyz / max(m, 1e-6)
    return xyz.astype(np.float32)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    H = src_c.T @ dst_c / n
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = U @ Vt
    var_src = (src_c**2).sum() / n
    s = 1.0 if var_src < 1e-12 else float(np.trace(np.diag(S)) / var_src)
    t = mu_d - s * (mu_s @ R.T)
    return R.astype(np.float64), s, t.astype(np.float64)


def apply_similarity(xyz: np.ndarray, R: np.ndarray, s: float, t: np.ndarray) -> np.ndarray:
    return (s * (xyz @ R.T) + t).astype(np.float32)


def icp_umeyama_align(
    pv_xyz: np.ndarray, obj_xyz: np.ndarray, n_iter: int
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, float, float]:
    pv = np.asarray(pv_xyz, dtype=np.float64)
    obj = np.asarray(obj_xyz, dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    s = 1.0
    t = np.zeros(3, dtype=np.float64)
    for _ in range(n_iter):
        pv_t = apply_similarity(pv, R, s, t).astype(np.float64)
        ij = nn_argmin(obj, pv_t.astype(np.float32)).astype(np.int64)
        src = pv[ij]
        R, s, t = umeyama_similarity(src, obj)
    pv_al = apply_similarity(pv, R, s, t)
    ij_final = nn_argmin(obj_xyz, pv_al)
    gap = np.linalg.norm(obj_xyz - pv_al[ij_final], axis=1)
    return R, s, t, pv_al, float(np.sqrt((gap**2).mean())), float(gap.max())


def load_group_ids_from_pack(pack: np.lib.npyio.NpzFile, k: int) -> np.ndarray:
    key_r = f"patch_r{k}"
    key_map = f"map64_to_{k}"
    if key_r in pack:
        return pack[key_r].astype(np.int32)
    if "patch_r64" in pack and key_map in pack:
        return pack[key_map][pack["patch_r64"]].astype(np.int32)
    raise KeyError(f"Pack has no labels for K={k}")


@dataclass
class SemanticPartStats:
    semantic_id: int
    group_ids: List[int]
    union_iou: float
    n_semantic_points: int
    n_union_points: int


def masks_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


PointPatchAssignMode = Literal["pointbert_group", "voronoi"]


def primary_patch_per_point_voronoi(pc_xyz: np.ndarray, patch_xyz: np.ndarray) -> np.ndarray:
    """One patch per point = nearest FPS center (legacy; discrete Voronoi on patch_xyz)."""
    d2 = np.sum((pc_xyz[:, None, :] - patch_xyz[None, :, :]) ** 2, axis=-1)
    return np.argmin(d2, axis=1).astype(np.int64)


def primary_patch_per_point_pointbert_group(
    pc_xyz: np.ndarray,
    patch_xyz: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """
    Match PointBERT ``Group``: for each patch g, neighborhood = ``group_size`` closest points to
    center ``patch_xyz[g]``. Inverse map: point i → primary patch among those that include i in their
    neighborhood, breaking ties by smallest squared distance to that patch's center; if none,
    Voronoi fallback (nearest center).
    """
    n_pts, _ = pc_xyz.shape
    n_patch = int(patch_xyz.shape[0])
    k_nn = min(int(group_size), n_pts)
    d2 = np.sum((pc_xyz[:, None, :] - patch_xyz[None, :, :]) ** 2, axis=-1)

    owners: List[List[int]] = [[] for _ in range(n_pts)]
    for g in range(n_patch):
        col = d2[:, g]
        idx = np.argpartition(col, k_nn - 1)[:k_nn]
        for pi in idx:
            owners[int(pi)].append(g)

    primary = np.empty(n_pts, dtype=np.int64)
    for i in range(n_pts):
        cand = owners[i]
        if cand:
            primary[i] = min(cand, key=lambda gg: float(d2[i, gg]))
        else:
            primary[i] = int(np.argmin(d2[i]))
    return primary


def primary_patch_per_point(
    pc_xyz: np.ndarray,
    patch_xyz: np.ndarray,
    mode: PointPatchAssignMode,
    pointbert_group_size: int,
) -> np.ndarray:
    if mode == "voronoi":
        return primary_patch_per_point_voronoi(pc_xyz, patch_xyz)
    if mode == "pointbert_group":
        return primary_patch_per_point_pointbert_group(
            pc_xyz, patch_xyz, pointbert_group_size
        )
    raise ValueError(f"Unknown point_patch_assign mode: {mode}")


def compute_object_mapping(
    obj_xyz: np.ndarray,
    pv_xyz: np.ndarray,
    point_part_id: np.ndarray,
    patch_xyz: np.ndarray,
    group_ids: np.ndarray,
    k_groups: int,
    icp_iter: int,
    min_group_iou: float,
    min_semantic_points: int,
    point_patch_assign: PointPatchAssignMode = "pointbert_group",
    pointbert_group_size: int = 32,
) -> Dict:
    N = obj_xyz.shape[0]
    R, s, t, pv_al, align_rmse, align_max = icp_umeyama_align(pv_xyz, obj_xyz, icp_iter)

    idx = nn_argmin(obj_xyz, pv_al)
    sem_per_point = point_part_id[idx].astype(np.int32)
    patch_idx = primary_patch_per_point(
        np.ascontiguousarray(obj_xyz[:, :3], dtype=np.float32),
        np.ascontiguousarray(patch_xyz, dtype=np.float32),
        point_patch_assign,
        pointbert_group_size,
    )
    grp_per_point = group_ids[patch_idx.astype(np.int32, copy=False)]

    valid_sem = sem_per_point >= 0
    group_masks = [grp_per_point == g for g in range(k_groups)]

    sem_masks: Dict[int, np.ndarray] = {}
    for pid in np.unique(sem_per_point[valid_sem]):
        sem_masks[int(pid)] = np.logical_and(sem_per_point == pid, valid_sem)

    group_to_sem: Dict[int, int] = {}
    group_best_iou: Dict[int, float] = {}
    for g in range(k_groups):
        gm = group_masks[g]
        if gm.sum() == 0:
            continue
        best_p, best_iou = -1, -1.0
        for pid, sm in sem_masks.items():
            iou = masks_iou(gm, sm)
            if iou > best_iou:
                best_iou, best_p = iou, pid
        if best_iou >= min_group_iou and best_p >= 0:
            group_to_sem[g] = best_p
            group_best_iou[g] = best_iou

    sem_to_groups: Dict[int, List[int]] = {}
    for g, p in group_to_sem.items():
        sem_to_groups.setdefault(p, []).append(g)
    for p in sem_to_groups:
        sem_to_groups[p] = sorted(sem_to_groups[p])

    per_sem: List[SemanticPartStats] = []
    for pid, sm in sem_masks.items():
        if sm.sum() < min_semantic_points:
            continue
        gs = sem_to_groups.get(pid, [])
        if not gs:
            per_sem.append(
                SemanticPartStats(int(pid), [], 0.0, int(sm.sum()), 0)
            )
            continue
        union_m = np.zeros(N, dtype=bool)
        for g in gs:
            union_m |= group_masks[g]
        ui = masks_iou(sm, union_m)
        per_sem.append(
            SemanticPartStats(int(pid), gs, ui, int(sm.sum()), int(union_m.sum()))
        )

    per_sem.sort(key=lambda x: x.semantic_id)
    ious = [x.union_iou for x in per_sem if x.group_ids]
    return {
        "point_patch_assign": point_patch_assign,
        "pointbert_group_size": int(pointbert_group_size),
        "icp_iter": icp_iter,
        "align_rmse": align_rmse,
        "align_max_error": align_max,
        "similarity_R": R.tolist(),
        "similarity_s": s,
        "similarity_t": t.tolist(),
        "group_to_semantic": {str(k): int(v) for k, v in sorted(group_to_sem.items())},
        "semantic_to_groups": {str(k): v for k, v in sorted(sem_to_groups.items())},
        "group_best_iou": {str(k): float(v) for k, v in sorted(group_best_iou.items())},
        "per_semantic": [asdict(x) for x in per_sem],
        "mean_union_iou_nonempty": float(np.mean(ious)) if ious else 0.0,
        "num_semantic_parts_labeled": len(sem_masks),
        "num_semantic_with_groups": sum(1 for x in per_sem if x.group_ids),
    }


def load_inputs(
    obj_id: str,
    objaverse_dir: Path,
    partverse_dir: Path,
    pack_path: Path,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj_npy = objaverse_dir / f"{obj_id}_8192.npy"
    if not obj_npy.exists():
        raise FileNotFoundError(f"Missing Objaverse cloud: {obj_npy}")
    pv_npz = partverse_dir / f"{obj_id}.npz"
    if not pv_npz.exists():
        raise FileNotFoundError(f"Missing PartVerse npz: {pv_npz}")
    if not pack_path.exists():
        raise FileNotFoundError(f"Missing pack: {pack_path}")

    pc = np.load(obj_npy)
    obj_xyz = pc_norm_numpy(pc)
    with np.load(pv_npz) as pv:
        pv_xyz = pv["global_xyz"].astype(np.float32)
        point_part_id = pv["point_part_id"].astype(np.int32)
    with np.load(pack_path) as pack:
        patch_xyz = pack["patch_xyz"].astype(np.float32)
        group_ids = load_group_ids_from_pack(pack, k)

    if pv_xyz.shape[0] != obj_xyz.shape[0]:
        raise ValueError(
            f"Point count mismatch obj={obj_xyz.shape[0]} partverse={pv_xyz.shape[0]} for {obj_id}"
        )
    return obj_xyz, pv_xyz, point_part_id, patch_xyz, group_ids


def cmd_verify(args: argparse.Namespace) -> None:
    tup = load_inputs(
        args.obj_id,
        Path(args.objaverse_dir),
        Path(args.partverse_dir),
        Path(args.bcp_pack_dir) / f"{args.obj_id}.pack.npz",
        args.k,
    )
    out = compute_object_mapping(
        *tup,
        k_groups=args.k,
        icp_iter=args.icp_iter,
        min_group_iou=args.min_group_iou,
        min_semantic_points=args.min_semantic_points,
        point_patch_assign=args.point_patch_assign,
        pointbert_group_size=args.pointbert_group_size,
    )
    print(json.dumps({"object_id": args.obj_id, "k": args.k, **out}, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    pv_dir = Path(args.partverse_dir)
    pack_dir = Path(args.bcp_pack_dir)
    obj_dir = Path(args.objaverse_dir)

    pv_ids = {p.stem for p in pv_dir.glob("*.npz")}
    pack_ids = {p.name.replace(".pack.npz", "") for p in pack_dir.glob("*.pack.npz")}
    common = sorted(pv_ids & pack_ids)
    if getattr(args, "shuffle_seed", None) is not None:
        rng = np.random.default_rng(int(args.shuffle_seed))
        rng.shuffle(common)
    if args.limit:
        common = common[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip = 0, 0
    hist_union: List[float] = []

    with open(out_path, "w", encoding="utf-8", buffering=1) as f:
        for oid in common:
            if not (obj_dir / f"{oid}_8192.npy").exists():
                n_skip += 1
                continue
            try:
                tup = load_inputs(oid, obj_dir, pv_dir, pack_dir / f"{oid}.pack.npz", args.k)
                rec = compute_object_mapping(
                    *tup,
                    k_groups=args.k,
                    icp_iter=args.icp_iter,
                    min_group_iou=args.min_group_iou,
                    min_semantic_points=args.min_semantic_points,
                    point_patch_assign=args.point_patch_assign,
                    pointbert_group_size=args.pointbert_group_size,
                )
            except Exception as e:
                n_skip += 1
                if args.verbose:
                    print(f"SKIP {oid}: {e}", file=sys.stderr)
                continue

            if rec["align_rmse"] > args.max_align_rmse:
                n_skip += 1
                if args.verbose:
                    print(
                        f"SKIP {oid}: align_rmse {rec['align_rmse']:.4f} > {args.max_align_rmse}",
                        file=sys.stderr,
                    )
                continue

            row = {"object_id": oid, "k": args.k, **rec}
            for ps in rec["per_semantic"]:
                if ps["group_ids"]:
                    hist_union.append(ps["union_iou"])
            f.write(json.dumps(row) + "\n")
            n_ok += 1

    print(f"Wrote {n_ok} records to {out_path}, skipped {n_skip}")
    if hist_union:
        h = np.array(hist_union)
        print(
            "Union IoU (semantic with groups): "
            f"mean={h.mean():.3f} p50={np.median(h):.3f} p10={np.percentile(h, 10):.3f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="PartVerse <-> BCP semantic mapping")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--partverse_dir", type=str, default=PARTVERSE_POINT_CACHE)
        sp.add_argument("--bcp_pack_dir", type=str, default=BCP_PACK_DIR)
        sp.add_argument("--objaverse_dir", type=str, default=OBJAVERSE_8192_DIR)
        sp.add_argument("--k", type=int, default=16)
        sp.add_argument("--icp_iter", type=int, default=12)
        sp.add_argument("--min_group_iou", type=float, default=0.02)
        sp.add_argument("--min_semantic_points", type=int, default=32)
        sp.add_argument(
            "--point_patch_assign",
            type=str,
            choices=("pointbert_group", "voronoi"),
            default="pointbert_group",
            help="How each Objaverse point is assigned a patch before reading patch_rK (default: "
            "PointBERT Group–consistent; voronoi = legacy nearest-center).",
        )
        sp.add_argument(
            "--pointbert_group_size",
            type=int,
            default=32,
            help="Must match PointBERT group_size used when building packs (BCP v3 uses 32).",
        )

    sp_v = sub.add_parser("verify", help="One object JSON to stdout")
    add_common(sp_v)
    sp_v.add_argument("--obj_id", type=str, required=True)
    sp_v.set_defaults(func=cmd_verify)

    sp_b = sub.add_parser("build", help="Batch JSONL")
    add_common(sp_b)
    sp_b.add_argument(
        "--out",
        type=str,
        default=str(Path(OUTPUT_DIR) / "mappings_k16.jsonl"),
        help="Output JSONL",
    )
    sp_b.add_argument("--limit", type=int, default=0)
    sp_b.add_argument("--max_align_rmse", type=float, default=0.08)
    sp_b.add_argument("--verbose", action="store_true")
    sp_b.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="If set, shuffle object ids before --limit (avoids lexicographic bias).",
    )
    sp_b.set_defaults(func=cmd_build)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
