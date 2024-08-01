
if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

MODEL_NAME=""
STEP_NUM=
ACTOR_MODEL_NAME_OR_PATH=MCTS-DPO/outputs/checkpoints/sqa/${MODEL_NAME}/steps${STEP_NUM}

OUTPUT_DIR="MCTS-DPO/outputs/eval"
unset HOSTFILE
ZERO_STAGE=2
OFFLOAD="all"


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

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P


gpu_vis=$1


deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT \
	--module mcts_rl.algorithms.mcts \
	--train_datasets SQA/train \
	--eval_datasets MCQ/test \
	--actor_model_name_or_path "${ACTOR_MODEL_NAME_OR_PATH}" \
	--actor_ref_model_name_or_path "${ACTOR_MODEL_NAME_OR_PATH}" \
	--max_length 512 \
	--repetition_penalty 1.0 \
	--trust_remote_code True \
	--epochs 1 \
	--update_iters 1 \
	--save_interval 128 \
	--per_device_prompt_batch_size 1 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps 8 \
	--actor_lr 1e-6 \
	--actor_weight_decay 0.01 \
	--actor_lr_scheduler_type cosine \
	--actor_lr_warmup_ratio 0.03 \
	--actor_gradient_checkpointing \
	--need_eval \
	--seed 42 \
	--kl_coeff 0.02 \
	--clip_range_ratio 0.2 \
	--clip_range_score 50.0 \
	--clip_range_value 5.0 \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project MCTS-DPO-EVAL \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 True \
	--tf32 True \
	--max_new_tokens 32 \
	--depth_limit 3 \
	--n_init_actions 4 \
	--n_actions 3 \
	--n_iters 16 \
	--mcts_temperature 0.0 \
	--num_return_sequences 1 \
	--temperature 1.0 \
	--init_temperature 1.0 \
	--model_type mistral \
	--use_mcq \
	--prediction_file_path MCTS-DPO/outputs/checkpoints/sqa/predictions/mistral_${MODEL_NAME}_${STEP_NUM}.jsonl
