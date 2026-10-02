#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
set -euo pipefail

master_port=${MASTER_PORT:-$((RANDOM % (65535 - 49152 + 1) + 49152))}
filename=$(basename "$0" .sh)

project_root="${PLOT_ROOT}"
PYTHON="${PLOT_PYTHON}"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$project_root:${PYTHONPATH:-}"
export TORCH_COMPILE_DISABLE=1

########################################
# Paths
########################################

model_name_or_path="${POINTLLM_CKPT_DIR}/PointLLM_7B_v1.1_init"
data_path="${POINTLLM_DATA}/objaverse_data"
anno_path="${POINTLLM_DATA}/anno_data/PointLLM_brief_description_660K_filtered.json"
output_dir="${project_root}/outputs/PointLLM_train_stage1/$filename"
point_backbone_ckpt="${model_name_or_path}/point_bert_v1.2.pt"

bcp_pack_dir="${BCP_PACK_DIR}"

mkdir -p "$output_dir"

########################################
# Auto-resume
########################################

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

########################################
# Launch (2 GPUs, DDP)
########################################

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
  \
  --num_train_epochs 3 \
  --per_device_train_batch_size 24 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 300 \
  --save_total_limit 2 \
  \
  --learning_rate 2e-3 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 20 \
  \
  --bf16 True \
  --fix_llm True \
  --fix_pointnet True \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  \
  --report_to none \
  --run_name "${filename}" \
  \
  --point_backbone_ckpt "${point_backbone_ckpt}" \
  --use_color True \
  \
  --bcp_pack_dir "${bcp_pack_dir}" \
  --bcp_part_interleave True \
  --bcp_part_marker per_part \
  --bcp_part_no_mean True \
  ${resume_arg} \
  "$@"

########################################
