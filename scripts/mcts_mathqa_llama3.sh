if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

ACTOR_MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
ACTOR_REF_MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B-Instruct"

OUTPUT_DIR="MCTS-DPO/outputs/checkpoints/arithmetic/llama3-cdpo-2x2-gtsft"
unset HOSTFILE
ZERO_STAGE=3
OFFLOAD="optimizer"


mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

export WANDB_API_KEY=""
export WANDB_MODE=online
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

gpu_vis=$1

deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT \
	--module mcts_rl.algorithms.mcts \
	--train_datasets MathQA/train \
	--model_type llama3 \
	--choose_worst \
	--save_mcts_data \
	--filter \
	--iteration_interval 64 \
	--actor_model_name_or_path "${ACTOR_MODEL_NAME_OR_PATH}" \
	--actor_ref_model_name_or_path "${ACTOR_REF_MODEL_NAME_OR_PATH}" \
	--scale_coeff 0.1 \
	--max_length 512 \
	--temperature 1.0 \
	--init_temperature 1.0 \
	--mcts_length_penalty 1.25 \
	--num_return_sequences 1 \
	--repetition_penalty 1.0 \
	--trust_remote_code True \
	--epochs 1 \
	--conservative \
	--update_iters 1 \
	--save_interval 64 \
	--per_device_ptx_batch_size 4 \
	--per_device_prompt_batch_size 1 \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 64 \
	--actor_lr 1e-6 \
	--actor_weight_decay 0.05 \
	--actor_lr_scheduler_type cosine \
	--actor_lr_warmup_ratio 0.03 \
	--actor_gradient_checkpointing \
	--seed 42 \
	--kl_coeff 0.02 \
	--clip_range_ratio 0.2 \
	--clip_range_score 50.0 \
	--clip_range_value 5.0 \
	--ptx_coeff 0.0 \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project MCTS-IPL-Math \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 True \
	--tf32 True \
	--max_new_tokens 128 \
	--n_iters 64 \
	--depth_limit 3 \
	--n_init_actions 2 \
	--n_actions 2 \
	--force_terminating_on_depth_limit \
	--mcts_temperature 0.0