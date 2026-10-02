#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# ============================================================================
# PartVerse-QA evaluation for every checkpoint in the MODELS list below:
#   caption-to-slots (greedy, one run; Jaccard / exact match computed by the eval script)
#   slots-to-caption (sampling, 5 runs) + traditional metrics + aggregation
# Writes under <ckpt>/evaluation/. No GPT API calls.
#
# Usage:
#   bash eval_partverse.sh              # run all listed models
#   bash eval_partverse.sh --dry-run    # print the commands only
#   GPU_IDS=0,1 bash eval_partverse.sh  # two models in parallel on two GPUs
#
# Evaluation files default to $PARTVERSE_QA_DIR/eval_c2s.json and eval_s2c.json; override with
#   PARTVERSE_ANNO_CAPTION2SLOTS=/path/to/eval_c2s_one_per_object.json
#   PARTVERSE_ANNO_SLOTS2CAPTION=/path/to/eval_s2c_one_per_object.json
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

data_path="${POINTLLM_DATA}/objaverse_data"
bcp_pack_dir="${BCP_PACK_DIR}"

if [[ ! -d "${PARTVERSE_QA_DIR}" ]]; then
    echo "PartVerse-QA directory not found: ${PARTVERSE_QA_DIR} (set PARTVERSE_QA_DIR)" >&2
    exit 1
fi
anno_caption2slots="${PARTVERSE_ANNO_CAPTION2SLOTS:-${PARTVERSE_QA_DIR}/eval_c2s.json}"
anno_slots2caption="${PARTVERSE_ANNO_SLOTS2CAPTION:-${PARTVERSE_QA_DIR}/eval_s2c.json}"

echo "PartVerse-QA C2S queries: ${anno_caption2slots}"
echo "PartVerse-QA S2C queries: ${anno_slots2caption}"

# ============================================================================
# MODEL REGISTRY
# Format: "SHORT_NAME | CHECKPOINT_PATH | PACK_DIR_OR_NONE"
# Models with "NONE" (no partition packs) are skipped; so are missing checkpoints.
# ============================================================================
MODELS=(
    "3D-PLOT-LLM-7B | ${POINTLLM_CKPT_DIR}/3D-PLOT-LLM-7B | ${bcp_pack_dir}"
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

total=${#MODELS[@]}
idx=0
failed_models=()
# Serial on one GPU by default; GPU_IDS=0,1 runs two models at a time.
GPU_IDS_CSV="${GPU_IDS:-0}"
IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS_CSV"
if [[ ${#GPU_IDS_ARR[@]} -eq 0 ]]; then
    GPU_IDS_ARR=("0")
fi
MAX_JOBS=${#GPU_IDS_ARR[@]}

echo "Using GPUs: ${GPU_IDS_ARR[*]} (parallel jobs=${MAX_JOBS})"

run_one_model() {
    local name="$1"
    local ckpt_path="$2"
    local bcp_dir="$3"
    local gpu_id="$4"

    echo ""
    echo "============================================================"
    echo " [GPU ${gpu_id}] ${name} (PartVerse-QA)"
    echo "   Checkpoint: ${ckpt_path}"
    echo "============================================================"

    local eval_dir="${ckpt_path}/evaluation"
    mkdir -p "$eval_dir"

    local base_cap
    local out_cap
    local base_s2c
    base_cap=$(basename "$anno_caption2slots" .json)
    out_cap="${eval_dir}/${base_cap}_partverse_caption2slots_pred.json"
    base_s2c=$(basename "$anno_slots2caption" .json)

    # ---- caption-to-slots (greedy, single run) ----
    if [[ -f "$out_cap" ]]; then
        echo "  >> [GPU ${gpu_id}] caption-to-slots exists, skip: ${out_cap}"
    else
        local CMD_CAP="$PYTHON_INFER pointllm/eval/eval_partverse_caption2slots.py \
            --model_name ${ckpt_path} \
            --anno_path ${anno_caption2slots} \
            --data_path ${data_path} \
            --bcp_pack_dir ${bcp_dir}"
        if $DRY_RUN; then
            echo "  [DRY-RUN][GPU ${gpu_id}] CUDA_VISIBLE_DEVICES=${gpu_id} $CMD_CAP"
        else
            if ! CUDA_VISIBLE_DEVICES="${gpu_id}" eval "$CMD_CAP"; then
                echo "  [ERROR] [GPU ${gpu_id}] caption-to-slots failed for ${name}"
                return 1
            fi
        fi
    fi

    # ---- slots-to-caption (sampling, 5 runs; same decoding as Objaverse captioning) ----
    local s2c_temperature="${S2C_TEMPERATURE:-1.0}"
    local s2c_top_p="${S2C_TOP_P:-0.95}"
    local s2c_top_k="${S2C_TOP_K:-50}"
    local s2c_num_runs="${S2C_NUM_RUNS:-5}"
    for run_id in $(seq 1 "${s2c_num_runs}"); do
        local out_s2c_run="${eval_dir}/${base_s2c}_partverse_slots2caption_pred_run${run_id}.json"
        if [[ -f "$out_s2c_run" ]]; then
            echo "  >> [GPU ${gpu_id}] slots-to-caption run ${run_id} exists, skip: ${out_s2c_run}"
        else
            # --run_id only names the output file; sampling is unseeded.
            local CMD_S2C="$PYTHON_INFER pointllm/eval/eval_partverse_slots2caption.py \
                --model_name ${ckpt_path} \
                --anno_path ${anno_slots2caption} \
                --data_path ${data_path} \
                --bcp_pack_dir ${bcp_dir} \
                --temperature ${s2c_temperature} \
                --top_p ${s2c_top_p} \
                --top_k ${s2c_top_k} \
                --run_id ${run_id}"
            if $DRY_RUN; then
                echo "  [DRY-RUN][GPU ${gpu_id}] CUDA_VISIBLE_DEVICES=${gpu_id} $CMD_S2C"
            else
                if ! CUDA_VISIBLE_DEVICES="${gpu_id}" eval "$CMD_S2C"; then
                    echo "  [ERROR] [GPU ${gpu_id}] slots-to-caption run ${run_id} failed for ${name}"
                    return 1
                fi
            fi
        fi

        # traditional metrics for this run
        local trad_json="${eval_dir}/${base_s2c}_partverse_slots2caption_pred_run${run_id}_evaluated_traditional.json"
        if [[ -f "$trad_json" ]]; then
            echo "  >> [GPU ${gpu_id}] traditional metrics run ${run_id} exist, skip."
        elif [[ -f "$out_s2c_run" || "$DRY_RUN" == true ]]; then
            local CMD_TRAD="$PYTHON_EVAL pointllm/eval/traditional_evaluator.py \
                --results_path ${out_s2c_run} \
                --output_dir ${eval_dir}"
            if $DRY_RUN; then
                echo "  [DRY-RUN][GPU ${gpu_id}] $CMD_TRAD"
            else
                if ! eval "$CMD_TRAD"; then
                    echo "  [ERROR] [GPU ${gpu_id}] traditional metrics run ${run_id} failed for ${name}"
                    return 1
                fi
            fi
        fi
    done

    # ---- aggregate -> <ckpt>/evaluation/_aggregate_partverse.json ----
    local CMD_AGG="PLOT_EVAL_ROOT=$(dirname "$ckpt_path") $PYTHON_EVAL eval_tools/aggregate.py $(basename "$ckpt_path")"
    if $DRY_RUN; then
        echo "  [DRY-RUN][GPU ${gpu_id}] $CMD_AGG"
    else
        eval "$CMD_AGG" || echo "  [WARN] aggregation failed for ${name}"
    fi
    return 0
}

running_pids=()
running_names=()
running_gpus=()
gpu_cursor=0

wait_one_job() {
    local pid="${running_pids[0]}"
    local name="${running_names[0]}"
    local gpu="${running_gpus[0]}"
    local rc=0
    if ! wait "$pid"; then
        rc=$?
    fi
    running_pids=("${running_pids[@]:1}")
    running_names=("${running_names[@]:1}")
    running_gpus=("${running_gpus[@]:1}")
    if [[ $rc -ne 0 ]]; then
        echo ">>> [FAILED][GPU ${gpu}] ${name}"
        failed_models+=("${name}")
    else
        echo ">>> [DONE][GPU ${gpu}] ${name}"
    fi
}

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name ckpt_path bcp_dir <<< "$entry"
    name=$(echo "$name" | xargs)
    ckpt_path=$(echo "$ckpt_path" | xargs)
    bcp_dir=$(echo "$bcp_dir" | xargs)

    idx=$((idx + 1))

    if [[ "$bcp_dir" == "NONE" ]]; then
        echo ">>> [${idx}/${total}] SKIP ${name} (no partition packs)"
        continue
    fi
    if [[ ! -d "$ckpt_path" ]]; then
        echo ">>> [${idx}/${total}] SKIP ${name} (checkpoint not found: ${ckpt_path})"
        continue
    fi

    gpu_id="${GPU_IDS_ARR[$gpu_cursor]}"
    gpu_cursor=$(( (gpu_cursor + 1) % MAX_JOBS ))

    echo ">>> [${idx}/${total}] dispatch ${name} -> GPU ${gpu_id}"
    if $DRY_RUN; then
        run_one_model "$name" "$ckpt_path" "$bcp_dir" "$gpu_id"
        continue
    fi

    run_one_model "$name" "$ckpt_path" "$bcp_dir" "$gpu_id" &
    running_pids+=("$!")
    running_names+=("$name")
    running_gpus+=("$gpu_id")

    while [[ ${#running_pids[@]} -ge ${MAX_JOBS} ]]; do
        wait_one_job
    done
done

while [[ ${#running_pids[@]} -gt 0 ]]; do
    wait_one_job
done

echo ""
echo "============================================================"
echo " PartVerse-QA evaluation done. Mesh mIoU: eval_tools/compute_mesh_miou.py --pred <c2s_pred.json>"
if [[ ${#failed_models[@]} -gt 0 ]]; then
    echo " Failed models (${#failed_models[@]}): ${failed_models[*]}"
fi
echo "============================================================"
