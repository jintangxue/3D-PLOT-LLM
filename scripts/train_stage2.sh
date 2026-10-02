#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
# Stage 2 of the full model: <part_k> vocabulary tokens + markers + MSR, trained on PointLLM 70K + PartVerse-QA.
set -euo pipefail

master_port=${MASTER_PORT:-$((RANDOM % (65535 - 49152 + 1) + 49152))}
filename=$(basename "$0" .sh)

project_root="${PLOT_ROOT}"
PYTHON="${PLOT_PYTHON}"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$project_root:${PYTHONPATH:-}"
export TORCH_COMPILE_DISABLE=1

model_name_or_path="${project_root}/outputs/PointLLM_train_stage1/train_stage1"
data_path="${POINTLLM_DATA}/objaverse_data"
anno_path="${PARTVERSE_QA_DIR}/train_stage2.json"
output_dir="${project_root}/outputs/PointLLM_train_stage2/$filename"

bcp_pack_dir="${BCP_PACK_DIR}"
ds_config="${project_root}/scripts/zero3.json"

mkdir -p "$output_dir"

latest_ckpt=""
if [ -d "$output_dir" ]; then
  latest_ckpt=$(ls -dt "$output_dir"/checkpoint-* 2>/dev/null | head -n 1 || true)
fi

resume_arg=""
if [ -n "${latest_ckpt}" ]; then
  echo ">>> Resuming from checkpoint: ${latest_ckpt}"
  resume_arg="--resume_from_checkpoint ${latest_ckpt}"
else
  echo ">>> No checkpoint found, starting from scratch."
fi

"$PYTHON" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_port="${master_port}" \
  pointllm/train/train_mem.py \
  --model_name_or_path "${model_name_or_path}" \
  --data_path "${data_path}" \
  --anno_path "${anno_path}" \
  --output_dir "${output_dir}" \
  --version v1 \
  --model_max_length 2048 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 300 \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 20 \
  --bf16 True \
  --fix_llm False \
  --fix_pointnet True \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --stage_2 True \
  --deepspeed "${ds_config}" \
  --tune_mm_mlp_adapter False \
  --conversation_types "detailed_description" "single_round" "multi_round" \
    "partverse_caption2slots" "partverse_slots2caption" \
  --report_to none \
  --run_name "${filename}" \
  --use_color True \
  --bcp_pack_dir "${bcp_pack_dir}" \
  --bcp_part_interleave True \
  --bcp_part_marker per_part \
  --bcp_part_no_mean True \
  --bcp_part_vocab_token True \
  --bcp_marker_context_mode stats_graph \
  ${resume_arg} \
  "$@"

