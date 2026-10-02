"""Set-level Jaccard + Exact-match for PartVerse-QA C2S.

Logic mirrors the reference impl in eval_partverse_caption2slots.py:
empty gold or empty pred -> Jaccard 0; exact match needs gold non-empty.

Accepts both pred-file conventions:
- conversation schema: {object_id, conversations:[{from, value}], meta}
- inference-script schema: {object_id, gold, prediction, ...}

Usage:
    python compute_c2s_metrics.py --gt data/heldout_c2s_392.json --pred your_pred.json
"""

import argparse, json, re

PART_RE = re.compile(r"<part_(\d+)>")


def parse_slots(text):
    return {int(m.group(1)) for m in PART_RE.finditer(text or "")}


def gt_slots(entry):
    # GT slot set comes from meta.bcp_group_ids (held-out file is always conversation schema).
    return set(int(g) for g in entry.get("meta", {}).get("bcp_group_ids", []))


def pred_slots(entry):
    # Conversation schema: parse the gpt-turn text.
    for c in entry.get("conversations", []) or []:
        if c.get("from") == "gpt":
            return parse_slots(c.get("value", ""))
    # Inference-script schema: {object_id, gold, prediction}.
    if "prediction" in entry:
        return parse_slots(entry.get("prediction", ""))
    return set()


def jaccard(a, b):
    # Empty gold or empty pred -> 0 (matches reference).
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def exact_match(a, b):
    # Empty gold (shouldn't happen on held-out) doesn't count as a match.
    return 1.0 if (a == b and a) else 0.0


def full_key(entry):
    m = entry.get("meta", {}) or {}
    k = (entry.get("object_id"),
         m.get("semantic_part_id"),
         m.get("caption_source"),
         m.get("caption_view_key"))
    # If pred is in inference-script schema (no meta), the full key collapses to (oid, None, None, None);
    # the caller falls back to object_id-only / positional matching in that case.
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    gts = json.load(open(args.gt))
    preds = json.load(open(args.pred))
    # Inference scripts wrap results as {summary, results}; unwrap if present.
    if isinstance(gts, dict) and "results" in gts:
        gts = gts["results"]
    if isinstance(preds, dict) and "results" in preds:
        preds = preds["results"]

    pred_idx = {full_key(p): p for p in preds}
    use_full_key = len(pred_idx) == len(preds)
    # Fallback for inference-script schema (no meta -> full keys collapse): index by object_id alone.
    pred_by_oid = {p["object_id"]: p for p in preds}
    use_oid_key = (not use_full_key) and (len(pred_by_oid) == len(preds))

    js, ems, missed = [], [], 0
    for i, g in enumerate(gts):
        if use_full_key:
            p = pred_idx.get(full_key(g))
        elif use_oid_key:
            p = pred_by_oid.get(g["object_id"])
        else:
            p = preds[i] if i < len(preds) else None
        if p is None:
            missed += 1
            continue
        gs, ps = gt_slots(g), pred_slots(p)
        js.append(jaccard(gs, ps))
        ems.append(exact_match(gs, ps))

    n = len(js)
    if not n:
        print("no matched queries"); return
    tag = f" (missed {missed})" if missed else ""
    print(f"n={n}{tag}")
    print(f"Jaccard:     {sum(js)/n:.4f}")
    print(f"Exact-match: {100*sum(ems)/n:.2f}%")


if __name__ == "__main__":
    main()
