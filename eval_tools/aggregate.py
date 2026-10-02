#!/usr/bin/env python3
"""Standalone aggregator over a tree of evaluation outputs (set PLOT_EVAL_ROOT or pass --root).

Produces three aggregate JSONs per variant from the per-run files:
  _aggregate_objaverse.json                       — 5-run mean/std of Obj traditional
  _aggregate_objaverse_gpt_<judge>.json           — 5-run mean/std of Obj GPT judge
  _aggregate_partverse.json                       — PV (P2S Jaccard from 1 file +
                                                    S2C 5-run aggregates)

Skips an aggregate if the per-run files for it are absent.

Usage:
    python aggregate.py <variant>           # one variant
    python aggregate.py --all               # every variant under the evaluation root
    python aggregate.py v1 v2 v3            # multiple
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(os.environ.get('PLOT_EVAL_ROOT', os.path.join(os.environ.get('PLOT_ROOT','.'), 'outputs/PointLLM_train_stage2')))

# Trad metrics we aggregate (Obj + PV S2C share the same set)
TRAD_METRICS = [
    'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4',
    'rouge-1', 'rouge-2', 'rouge-l',
    'meteor', 'sbert_similarity', 'simcse_similarity',
]

OBJ_TRAD_BASENAME = 'PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2'
PV_S2C_BASENAME = 'eval_s2c_partverse_slots2caption_pred'
PV_P2S_BASENAME = 'eval_c2s_partverse_caption2slots_pred'


def _mean_std(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    if len(values) == 1:
        return {'mean': round(values[0], 4), 'std': 0.0}
    return {
        'mean': round(statistics.mean(values), 4),
        'std': round(statistics.stdev(values), 4),
    }


def _read_overall(path):
    """Per-run trad files store metrics under d['overall_scores']."""
    d = json.load(open(path))
    return d.get('overall_scores') or {}


def aggregate_objaverse_trad(eval_dir: Path, model_name: str):
    runs = sorted(glob.glob(str(eval_dir / f'{OBJ_TRAD_BASENAME}_run*_evaluated_traditional.json')))
    if not runs:
        return None
    metrics = {m: [] for m in TRAD_METRICS}
    for f in runs:
        scores = _read_overall(f)
        for m in TRAD_METRICS:
            if m in scores:
                metrics[m].append(float(scores[m]))
    out = {
        'model_name': model_name,
        'task': 'Objaverse captioning (traditional metrics)',
        'n_runs': len(runs),
        'source_json_files': runs,
        'metrics': {m: _mean_std(v) for m, v in metrics.items() if v},
    }
    return out


def aggregate_objaverse_gpt(eval_dir: Path, model_name: str, judge: str = 'gpt-4o-2024-08-06'):
    runs = sorted(glob.glob(str(eval_dir / f'{OBJ_TRAD_BASENAME}_run*_evaluated_{judge}.json')))
    if not runs:
        return None
    avgs = []
    for f in runs:
        d = json.load(open(f))
        s = d.get('average_score')
        if s is not None:
            avgs.append(float(s))
    if not avgs:
        return None
    ms = _mean_std(avgs)
    return {
        'model_name': model_name,
        'task': 'Objaverse captioning (GPT judge)',
        'judge': judge,
        'n_runs': len(runs),
        'mean': ms['mean'],
        'std': ms['std'],
        'source_json_files': runs,
    }


def aggregate_partverse(eval_dir: Path, model_name: str):
    out = {'model_name': model_name,
           'task': 'PartVerse (caption2slots Jaccard + slots2caption Word F1)'}
    has_anything = False

    p2s_path = eval_dir / f'{PV_P2S_BASENAME}.json'
    if p2s_path.exists():
        d = json.load(open(p2s_path))
        # The pred JSON has a top-level summary block we want
        summ = d.get('summary') or {}
        # Source naming: pred-files use mean_jaccard, mean_geometric_iou_*;
        # the existing aggregate JSONs we mimic flatten these to
        # jaccard / geometric_iou_*.
        # NB: use explicit None checks rather than `or` chains, because the
        # MSR no-PV checkpoint legitimately produces 0.0 jaccard (model
        # cannot emit \partk{k} tokens without PV-trained vocabulary).
        def _coalesce(*keys):
            for k in keys:
                if summ.get(k) is not None:
                    return summ[k]
            return None
        out['caption2slots'] = {
            'jaccard': _coalesce('mean_jaccard', 'jaccard_tokens', 'jaccard'),
            'exact_set_match_rate': summ.get('exact_set_match_rate'),
            'num_samples': summ.get('num_samples') or len(d.get('results', [])),
            'geometric_iou_points': _coalesce('mean_geometric_iou_points', 'geometric_iou_points'),
            'geometric_iou_patches': _coalesce('mean_geometric_iou_patches', 'geometric_iou_patches'),
        }
        # Drop nones
        out['caption2slots'] = {k: v for k, v in out['caption2slots'].items() if v is not None}
        if out['caption2slots']:
            has_anything = True

    # PV S2C GPT-4o judge across runs
    s2c_gpt_runs = sorted(glob.glob(str(
        eval_dir / f'{PV_S2C_BASENAME}_run*_evaluated_gpt-4o-2024-08-06.json')))
    s2c_gpt_avgs = []
    for f in s2c_gpt_runs:
        d = json.load(open(f))
        s = d.get('average_score')
        if s is not None:
            s2c_gpt_avgs.append(float(s))

    s2c_runs = sorted(glob.glob(str(eval_dir / f'{PV_S2C_BASENAME}_run*.json')))
    s2c_runs = [f for f in s2c_runs if 'evaluated' not in os.path.basename(f)]
    if s2c_runs:
        wf1s = []
        ems = []
        for f in s2c_runs:
            d = json.load(open(f))
            summ = d.get('summary') or {}
            # Use explicit None checks; 0.0 is a valid value.
            wf1 = next((summ[k] for k in ('mean_word_f1', 'word_f1_mean', 'word_f1')
                        if summ.get(k) is not None), None)
            if wf1 is not None:
                wf1s.append(float(wf1))
            em = next((summ[k] for k in ('exact_norm_match_rate', 'exact_match')
                       if summ.get(k) is not None), None)
            if em is not None:
                ems.append(float(em))
        s2c_block = {}
        if wf1s:
            ms = _mean_std(wf1s)
            s2c_block['word_f1_mean'] = ms['mean']
            s2c_block['word_f1_std'] = ms['std']
        if ems:
            s2c_block['exact_match'] = round(statistics.mean(ems), 6)
        s2c_block['num_runs'] = len(s2c_runs)
        if s2c_gpt_avgs:
            ms = _mean_std(s2c_gpt_avgs)
            s2c_block['gpt_judge'] = {
                'judge': 'gpt-4o-2024-08-06',
                'mean': ms['mean'],
                'std': ms['std'],
                'num_runs': len(s2c_gpt_avgs),
            }
        # Trad metrics across runs
        trad_runs = sorted(glob.glob(str(eval_dir / f'{PV_S2C_BASENAME}_run*_evaluated_traditional.json')))
        if trad_runs:
            metrics = {m: [] for m in TRAD_METRICS}
            for f in trad_runs:
                scores = _read_overall(f)
                for m in TRAD_METRICS:
                    if m in scores:
                        metrics[m].append(float(scores[m]))
            s2c_block['traditional'] = {
                'num_runs': len(trad_runs),
                **{m: _mean_std(v) for m, v in metrics.items() if v},
            }
        out['slots2caption'] = s2c_block
        has_anything = True

    return out if has_anything else None


def _resolve_eval_dir(variant_dir: Path) -> Path | None:
    """Find evaluation/ for a variant. Convention is variant/evaluation, but
    some variants nest inference output under variant/checkpoint-NNN/evaluation
    (e.g., MSR p=0.75)."""
    direct = variant_dir / 'evaluation'
    if direct.is_dir():
        return direct
    # Fallback: pick the most recent checkpoint-* dir with an evaluation/
    candidates = sorted(variant_dir.glob('checkpoint-*/evaluation'))
    return candidates[-1] if candidates else None


def process(variant: str, judge: str = 'gpt-4o-2024-08-06', dry_run: bool = False):
    vd = ROOT / variant
    eval_dir = _resolve_eval_dir(vd)
    if eval_dir is None:
        print(f'  [{variant}] no evaluation/ dir (top-level or under checkpoint-*/), skipping')
        return
    if 'checkpoint-' in str(eval_dir):
        print(f'  [{variant}] using nested {eval_dir.relative_to(vd)}')
    written = []
    skipped = []

    # Obj trad
    out_path = eval_dir / '_aggregate_objaverse.json'
    res = aggregate_objaverse_trad(eval_dir, variant)
    if res:
        if not dry_run:
            json.dump(res, open(out_path, 'w'), indent=2)
        sb = res['metrics'].get('sbert_similarity', {}).get('mean')
        written.append(f'OBJ-trad ({res["n_runs"]} runs, SBERT={sb})')
    else:
        skipped.append('OBJ-trad')

    # Obj GPT
    out_path = eval_dir / f'_aggregate_objaverse_gpt_{judge}.json'
    res = aggregate_objaverse_gpt(eval_dir, variant, judge)
    if res:
        if not dry_run:
            json.dump(res, open(out_path, 'w'), indent=2)
        written.append(f'OBJ-GPT ({res["n_runs"]} runs, mean={res["mean"]})')
    else:
        skipped.append('OBJ-GPT')

    # PV
    out_path = eval_dir / '_aggregate_partverse.json'
    res = aggregate_partverse(eval_dir, variant)
    if res:
        if not dry_run:
            json.dump(res, open(out_path, 'w'), indent=2)
        c2s = res.get('caption2slots', {})
        s2c = res.get('slots2caption', {})
        written.append(f'PV (jac={c2s.get("jaccard")}, S2C wF1={s2c.get("word_f1_mean")})')
    else:
        skipped.append('PV')

    print(f'[{variant}]')
    for w in written:
        print(f'  ✓ {w}')
    for s in skipped:
        print(f'  - {s} (no per-run files)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variants', nargs='*', help='variant directory names under the evaluation root')
    ap.add_argument('--all', action='store_true', help='all directories under the evaluation root')
    ap.add_argument('--judge', default='gpt-4o-2024-08-06')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.all:
        variants = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    elif args.variants:
        variants = args.variants
    else:
        ap.print_help()
        sys.exit(1)

    for v in variants:
        process(v, judge=args.judge, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
