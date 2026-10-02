#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# ============================================================================
# Objaverse captioning: inference (5 sampled runs) + traditional metrics + aggregation
# for every checkpoint in the MODELS list below. No GPT API calls.
#
# Usage:
#   bash eval_objaverse.sh              # run all listed models
#   bash eval_objaverse.sh --dry-run    # print the commands only
# ============================================================================
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY-RUN] Will only print commands, not execute."
fi

project_root="${PLOT_ROOT}"
PYTHON_INFER="${PLOT_PYTHON}"
PYTHON_EVAL="${PLOT_PYTHON_EVAL}"
echo "PYTHON_INFER=$PYTHON_INFER"
echo "PYTHON_EVAL=$PYTHON_EVAL"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_COMPILE_DISABLE=1

# ---- Data paths ----
data_path="${POINTLLM_DATA}/objaverse_data"
anno_path="${POINTLLM_DATA}/anno_data/PointLLM_brief_description_val_200_GT.json"
bcp_pack_dir="${BCP_PACK_DIR}"

# ============================================================================
# MODEL REGISTRY
# Format: "SHORT_NAME | CHECKPOINT_PATH | PACK_DIR_OR_NONE"
# Use "NONE" for models without partition packs (the PointLLM baseline).
# Entries whose checkpoint directory does not exist are skipped.
# ============================================================================
MODELS=(
    "3D-PLOT-LLM-7B | ${POINTLLM_CKPT_DIR}/3D-PLOT-LLM-7B | ${bcp_pack_dir}"
    "PointLLM_7B_v1.2 | ${POINTLLM_CKPT_DIR}/PointLLM_7B_v1.2 | NONE"
    "train_stage2 | ${project_root}/outputs/PointLLM_train_stage2/train_stage2 | ${bcp_pack_dir}"
    "train_stage2_no_partverse | ${project_root}/outputs/PointLLM_train_stage2/train_stage2_no_partverse | ${bcp_pack_dir}"
    "stage2_vocab_only | ${project_root}/outputs/PointLLM_train_stage2/stage2_vocab_only | ${bcp_pack_dir}"
    "stage2_markers_no_msr | ${project_root}/outputs/PointLLM_train_stage2/stage2_markers_no_msr | ${bcp_pack_dir}"
    "stage2_lsr | ${project_root}/outputs/PointLLM_train_stage2/stage2_lsr | ${bcp_pack_dir}"
    "stage2_msr_plus_lsr | ${project_root}/outputs/PointLLM_train_stage2/stage2_msr_plus_lsr | ${bcp_pack_dir}"
    "stage2_msr_stats_only | ${project_root}/outputs/PointLLM_train_stage2/stage2_msr_stats_only | ${bcp_pack_dir}"
    "stage2_msr_adjacency_only | ${project_root}/outputs/PointLLM_train_stage2/stage2_msr_adjacency_only | ${bcp_pack_dir}"
    "stage2_markers_only_no_partverse | ${project_root}/outputs/PointLLM_train_stage2/stage2_markers_only_no_partverse | ${bcp_pack_dir}"
    "stage2_grouping_only_no_partverse | ${project_root}/outputs/PointLLM_train_stage2/stage2_grouping_only_no_partverse | ${bcp_pack_dir}"
    "stage2_partverse_16pct | ${project_root}/outputs/PointLLM_train_stage2/stage2_partverse_16pct | ${bcp_pack_dir}"
    "stage2_partverse_30pct | ${project_root}/outputs/PointLLM_train_stage2/stage2_partverse_30pct | ${bcp_pack_dir}"
    "stage2_partverse_50pct | ${project_root}/outputs/PointLLM_train_stage2/stage2_partverse_50pct | ${bcp_pack_dir}"
    "stage2_partverse_75pct | ${project_root}/outputs/PointLLM_train_stage2/stage2_partverse_75pct | ${bcp_pack_dir}"
    "stage2_k8 | ${project_root}/outputs/PointLLM_train_stage2/stage2_k8 | ${bcp_pack_dir}"
)

# ============================================================================
# MAIN LOOP: one model at a time. The 5 sampled runs share one python process
# (eval_objaverse.py --num_runs 5 skips runs whose _run{i}.json already exists).
# ============================================================================
total=${#MODELS[@]}
idx=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name ckpt_path bcp_dir <<< "$entry"
    name=$(echo "$name" | xargs)
    ckpt_path=$(echo "$ckpt_path" | xargs)
    bcp_dir=$(echo "$bcp_dir" | xargs)

    idx=$((idx + 1))
    if [[ ! -d "$ckpt_path" ]]; then
        echo ">>> [${idx}/${total}] SKIP ${name} (checkpoint not found: ${ckpt_path})"
        continue
    fi
    echo ""
    echo "============================================================"
    echo " [${idx}/${total}] ${name}"
    echo "   Checkpoint: ${ckpt_path}"
    echo "   Pack dir:   ${bcp_dir}"
    echo "============================================================"

    bcp_arg=""
    if [[ "$bcp_dir" != "NONE" ]]; then
        bcp_arg="--bcp_pack_dir ${bcp_dir}"
    fi

    eval_dir="${ckpt_path}/evaluation"
    anno_basename=$(basename "$anno_path" .json)

    # ---- Step 1: captioning inference (5 runs, batch size 1) ----
    all_exist=true
    for run_id in {1..5}; do
        rj="${eval_dir}/${anno_basename}_Objaverse_captioning_prompt2_run${run_id}.json"
        [[ -f "$rj" ]] || { all_exist=false; break; }
    done

    if $all_exist; then
        echo "  >> [1/3] All 5 inference runs already exist, skipping inference."
    else
        echo "  >> [1/3] Running captioning inference (5 runs in one process)..."
        CMD="$PYTHON_INFER pointllm/eval/eval_objaverse.py \
            --model_name ${ckpt_path} \
            --task_type captioning \
            --prompt_index 2 \
            --data_path ${data_path} \
            --anno_path ${anno_path} \
            --batch_size 1 \
            --num_workers 10 \
            --num_runs 5 \
            --run_start 1 \
            --force_run_suffix \
            ${bcp_arg}"
        if $DRY_RUN; then
            echo "  [DRY-RUN] $CMD"
        else
            eval "$CMD"
        fi
    fi

    # ---- Step 2: traditional metrics (BLEU / ROUGE / METEOR / SBERT / SimCSE) per run ----
    for run_id in {1..5}; do
        inference_json="${eval_dir}/${anno_basename}_Objaverse_captioning_prompt2_run${run_id}.json"
        trad_eval_json="${eval_dir}/${anno_basename}_Objaverse_captioning_prompt2_run${run_id}_evaluated_traditional.json"

        if [[ -f "$trad_eval_json" ]]; then
            echo "  >> [2/3] Traditional eval run ${run_id} exists, skipping."
        elif [[ ! -f "$inference_json" ]]; then
            echo "  >> [2/3] Inference run ${run_id} missing, skipping eval."
        else
            echo "  >> [2/3] Running traditional metrics run ${run_id}..."
            CMD="$PYTHON_EVAL pointllm/eval/traditional_evaluator.py \
                --results_path ${inference_json} \
                --output_dir ${eval_dir}"
            if $DRY_RUN; then
                echo "  [DRY-RUN] $CMD"
            else
                eval "$CMD"
            fi
        fi
    done

    # ---- Step 3: 5-run mean/std -> <ckpt>/evaluation/_aggregate_objaverse.json ----
    CMD="PLOT_EVAL_ROOT=$(dirname "$ckpt_path") $PYTHON_EVAL eval_tools/aggregate.py $(basename "$ckpt_path")"
    if $DRY_RUN; then
        echo "  [DRY-RUN] $CMD"
    else
        echo "  >> [3/3] Aggregating runs..."
        eval "$CMD" || echo "  [WARN] aggregation failed for ${name}"
    fi

    echo "  >> Done: ${name}"
done

echo ""
echo "============================================================"
echo " ALL MODELS PROCESSED. GPT-4o judge: pointllm/eval/evaluator.py (needs OPENAI_API_KEY)."
echo " PartVerse-QA: bash eval_partverse.sh"
echo "============================================================"
