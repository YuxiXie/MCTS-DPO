#!/usr/bin/env bash
#
# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

# MODEL_NAME_OR_PATH="facebook/opt-2.7b"
# MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-hf"
# MODEL_NAME_OR_PATH="mistralai/Mistral-7B-v0.1"
MODEL_NAME_OR_PATH="akjindal53244/Arithmo-Mistral-7B"
# MODEL_NAME_OR_PATH="/home/users/nus/e0672129/scratch/MCTS-DPO/sft/diymistral-arithmo/steps4202"
OUTPUT_DIR="/home/users/nus/e0672129/scratch/MCTS-DPO/sft/mcq"
unset HOSTFILE
ZERO_STAGE=3
OFFLOAD="optimizer"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

if [[ -z "${WANDB_API_KEY}" ]]; then
	export WANDB_MODE="offline"
fi

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

DEEPSPEED_ARGS=()
if [[ -n "${HOSTFILE+x}" ]]; then
	DEEPSPEED_ARGS+=("--hostfile" "${HOSTFILE}")
fi
DEEPSPEED_ARGS+=("--master_port" "${MASTER_PORT}")

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

export WANDB_API_KEY="1396a7d2a29a8e8241dff6e0e6371f2ad61e11e2"
export WANDB_MODE=online

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P

gpu_vis=0,1

# deepspeed "${DEEPSPEED_ARGS[@]}" \
deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT \
	--module mcts_rl.finetune \
	--train_datasets MCQ/train \
	--eval_datasets MCQ/test \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--max_length 256 \
	--trust_remote_code True \
	--epochs 3 \
	--per_device_train_batch_size 64 \
	--per_device_eval_batch_size 32 \
	--gradient_accumulation_steps 2 \
	--gradient_checkpointing \
	--learning_rate 5e-6 \
	--lr_scheduler_type cosine \
	--lr_warmup_ratio 0.03 \
	--weight_decay 0.0 \
	--seed 42 \
	--need_eval \
	--eval_strategy epoch \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project SFT-MCQ \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 True \
	--tf32 True