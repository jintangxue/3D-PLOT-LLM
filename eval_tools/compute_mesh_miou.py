#!/usr/bin/env python3
"""
Mesh-level partition-agnostic mIoU on PartVerse-QA C2S held-out (392 queries).

GT construction (matches bcp_semantic_mapping.compute_object_mapping):
  PV mesh-sampled cloud (cache_global_rgb_nn_partid/{obj}.global_xyz with
  point_part_id) is ICP+Umeyama aligned to PointLLM 8192 obj_xyz; alignments
  are pre-computed in alignment_k16.jsonl. Each PointLLM point
  takes the semantic id of its nearest aligned mesh point. gold_mask = points
  whose nearest-aligned semantic = queried semantic_part_id.

Pred mask: model's predicted <part_k> set -> K=16 patch groups -> 8192 points
via PointBERT-Group-style assignment (group_size=32), matching the eval
script's --point_patch_assign pointbert_group.

Reports per-query IoU(pred, mesh_gt), IoU(gold_slot_set, mesh_gt) (slot-set
ceiling), IoU(oracle, mesh_gt) (partition-oracle ceiling). All are
partition-agnostic in the GT side.
"""
import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ANNO_PATH = os.environ.get('PLOT_C2S_ANNO', os.path.join(os.environ.get('PARTVERSE_QA_DIR','data/partverse_qa'), 'eval_c2s.json'))
ALIGN_JSONL = os.path.join(os.environ.get('PARTVERSE_QA_DIR','data/partverse_qa'), 'alignment_k16.jsonl')
PC8192_DIR = os.path.join(os.environ.get('POINTLLM_DATA','data/pointllm'), 'objaverse_data')
SRP_PACK_DIR = os.environ.get('BCP_PACK_DIR','data/packs_k16')
PV_GLOBAL_DIR = os.environ.get('PARTVERSE_CACHE','data/partverse/cache_global_rgb_nn_partid')
GROUP_SIZE = 32  # matches pointbert_group_size


_ALIGN_CACHE = None


def load_alignments():
    global _ALIGN_CACHE
    if _ALIGN_CACHE is not None:
        return _ALIGN_CACHE
    out = {}
    with open(ALIGN_JSONL) as f:
        for line in f:
            d = json.loads(line)
            out[d['object_id']] = {
                'R': np.array(d['similarity_R'], dtype=np.float64),
                's': float(d['similarity_s']),
                't': np.array(d['similarity_t'], dtype=np.float64),
                'align_rmse': float(d.get('align_rmse', 0.0)),
            }
    _ALIGN_CACHE = out
    return out


def parse_slots(s):
    return {int(m) for m in re.findall(r'<part_(\d+)>', s, flags=re.IGNORECASE)}


def apply_similarity(xyz, R, s, t):
    return (s * (xyz.astype(np.float64) @ R.T) + t).astype(np.float32)


def pc_norm(pc_xyz):
    xyz = pc_xyz[:, :3].astype(np.float64)
    xyz = xyz - xyz.mean(axis=0)
    m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    xyz = xyz / max(m, 1e-6)
    return xyz.astype(np.float32)


def point_group_pointbert(pc_xyz, patch_xyz, patch_group, group_size=GROUP_SIZE):
    n_pts = pc_xyz.shape[0]
    n_patch = patch_xyz.shape[0]
    k_nn = min(int(group_size), n_pts)
    d2 = np.sum((pc_xyz[:, None, :] - patch_xyz[None, :, :]) ** 2, axis=-1)
    owners = [[] for _ in range(n_pts)]
    for g in range(n_patch):
        col = d2[:, g]
        idx = np.argpartition(col, k_nn - 1)[:k_nn]
        for pi in idx:
            owners[int(pi)].append(g)
    primary = np.empty(n_pts, dtype=np.int64)
    for i in range(n_pts):
        cand = owners[i]
        if cand:
            primary[i] = min(cand, key=lambda g: float(d2[i, g]))
        else:
            primary[i] = int(np.argmin(d2[i]))
    return patch_group[primary].astype(np.int64)


def build_mesh_sem_per_point(obj_xyz, pv_global_xyz, point_part_id, R, s, t):
    pv_al = apply_similarity(pv_global_xyz, R, s, t)
    tree = cKDTree(pv_al)
    _, idx = tree.query(obj_xyz)
    return point_part_id[idx].astype(np.int64)


def majority_oracle_slots(point_group, mesh_labels, sem_id, dominant_thresh=0.5):
    K = int(point_group[point_group >= 0].max()) + 1 if (point_group >= 0).any() else 0
    out = set()
    for c in range(K):
        cm = (point_group == c)
        cs = int(cm.sum())
        if cs == 0:
            continue
        labs = mesh_labels[cm]
        labs = labs[labs >= 0]
        if len(labs) == 0:
            continue
        u, cnts = np.unique(labs, return_counts=True)
        max_idx = int(np.argmax(cnts))
        if int(u[max_idx]) == sem_id and cnts[max_idx] / cs >= dominant_thresh:
            out.add(c)
    return out


def masks_iou(a, b):
    inter = int((a & b).sum())
    uni = int((a | b).sum())
    return float(inter / max(uni, 1))


_WORKER_ALIGN = None


def _init_worker():
    global _WORKER_ALIGN
    _WORKER_ALIGN = load_alignments()


def process_one(args):
    idx, anno_row, pred_row = args
    obj = anno_row['object_id']
    sem_id = int(anno_row['meta']['semantic_part_id'])
    gold_slots = parse_slots(anno_row['conversations'][1]['value'])
    pred_slots = parse_slots(pred_row['prediction'])

    pc_path = f'{PC8192_DIR}/{obj}_8192.npy'
    pack_path = f'{SRP_PACK_DIR}/{obj}.pack.npz'
    parts_path = f'{PV_GLOBAL_DIR}/{obj}.npz'
    align = (_WORKER_ALIGN or load_alignments()).get(obj)
    if align is None:
        return idx, {'__missing__': 'no_alignment'}
    if not all(os.path.exists(p) for p in [pc_path, pack_path, parts_path]):
        return idx, {'__missing__': 'no_files'}

    obj_xyz_raw = np.load(pc_path).astype(np.float32)
    obj_xyz = pc_norm(obj_xyz_raw)  # patch_xyz and saved alignment are both in normalized frame
    pv = np.load(parts_path)
    pv_global_xyz = pv['global_xyz'].astype(np.float32)
    point_part_id = pv['point_part_id'].astype(np.int64)

    mesh_sem = build_mesh_sem_per_point(
        obj_xyz, pv_global_xyz, point_part_id,
        align['R'], align['s'], align['t'],
    )
    gold_mask = (mesh_sem == sem_id)
    if int(gold_mask.sum()) < 50:
        return idx, {'__missing__': f'gold_too_small({int(gold_mask.sum())})'}

    pack = np.load(pack_path)
    patch_xyz = pack['patch_xyz'].astype(np.float32)
    valid = pack['patch_valid_mask'].astype(bool)
    patch_group = (pack['map64_to_16'][pack['patch_r64']]).astype(np.int64)
    patch_group[~valid] = -1

    point_group = point_group_pointbert(obj_xyz, patch_xyz, patch_group, GROUP_SIZE)

    pred_mask = np.isin(point_group, list(pred_slots)) if pred_slots else np.zeros_like(gold_mask)
    gold_slot_mask = np.isin(point_group, list(gold_slots)) if gold_slots else np.zeros_like(gold_mask)
    oracle_slots = majority_oracle_slots(point_group, mesh_sem, sem_id)
    oracle_mask = np.isin(point_group, list(oracle_slots)) if oracle_slots else np.zeros_like(gold_mask)

    return idx, {
        'object_id': obj,
        'sem_id': sem_id,
        'n_gold_pts': int(gold_mask.sum()),
        'pred_iou_mesh': masks_iou(pred_mask, gold_mask),
        'gold_slot_iou_mesh': masks_iou(gold_slot_mask, gold_mask),
        'oracle_iou_mesh': masks_iou(oracle_mask, gold_mask),
        'pred_slots': sorted(pred_slots),
        'gold_slots': sorted(gold_slots),
        'oracle_slots': sorted(oracle_slots),
        'align_rmse': align['align_rmse'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--out', default=None)
    parser.add_argument('--tag', default='')
    args = parser.parse_args()

    print(f'=== Mesh-level mIoU ({args.tag}) ===', flush=True)
    print(f'  pred: {args.pred}', flush=True)
    anno = json.load(open(ANNO_PATH))
    pred = json.load(open(args.pred))['results']
    assert len(anno) == len(pred), f'{len(anno)} vs {len(pred)}'
    items = [(i, a, p) for i, (a, p) in enumerate(zip(anno, pred))]

    out_recs = [None] * len(items)
    n_done = 0
    t0 = time.time()
    if args.workers <= 1:
        _init_worker()
        for it in items:
            i, rec = process_one(it)
            out_recs[i] = rec
            n_done += 1
    else:
        with Pool(args.workers, initializer=_init_worker) as pool:
            for i, rec in pool.imap_unordered(process_one, items, chunksize=4):
                out_recs[i] = rec
                n_done += 1
                if n_done % 80 == 0:
                    rate = n_done / (time.time() - t0)
                    print(f'    [{n_done}/{len(items)}] rate={rate:.1f}/s', flush=True)

    valid = [r for r in out_recs if r is not None and '__missing__' not in r]
    missing = [r for r in out_recs if r is not None and '__missing__' in r]
    miss_reasons = {}
    for r in missing:
        miss_reasons[r['__missing__']] = miss_reasons.get(r['__missing__'], 0) + 1
    print(f'\n  valid: {len(valid)}/{len(items)}  missing reasons: {miss_reasons}', flush=True)
    if not valid:
        return

    pred_iou = np.array([r['pred_iou_mesh'] for r in valid])
    gold_slot_iou = np.array([r['gold_slot_iou_mesh'] for r in valid])
    oracle_iou = np.array([r['oracle_iou_mesh'] for r in valid])

    print(f'\n  Model pred vs mesh GT     : mean={pred_iou.mean():.3f}  median={np.median(pred_iou):.3f}')
    print(f'  Gold slot set vs mesh GT  : mean={gold_slot_iou.mean():.3f}  median={np.median(gold_slot_iou):.3f}')
    print(f'  Partition oracle ceiling  : mean={oracle_iou.mean():.3f}  median={np.median(oracle_iou):.3f}')

    if args.out:
        json.dump({
            'pred_path': args.pred,
            'tag': args.tag,
            'n_total': len(items),
            'n_valid': len(valid),
            'mean_pred_iou_mesh': float(pred_iou.mean()),
            'median_pred_iou_mesh': float(np.median(pred_iou)),
            'mean_gold_slot_iou_mesh': float(gold_slot_iou.mean()),
            'median_gold_slot_iou_mesh': float(np.median(gold_slot_iou)),
            'mean_oracle_iou_mesh': float(oracle_iou.mean()),
            'median_oracle_iou_mesh': float(np.median(oracle_iou)),
            'per_query': valid,
            'missing_reasons': miss_reasons,
        }, open(args.out, 'w'), indent=2)
        print(f'\n  saved: {args.out}')


if __name__ == '__main__':
    sys.exit(main())
