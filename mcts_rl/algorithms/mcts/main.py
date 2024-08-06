"""The main training script to train RLHF using PPO algorithm."""

import argparse

import deepspeed
import torch
import torch.distributed as dist
from transformers import SchedulerType
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available

from mcts_rl.algorithms.mcts.trainer import MCTSTrainer
from mcts_rl.configs import get_deepspeed_eval_config, get_deepspeed_train_config
from mcts_rl.datasets import parse_dataset
from mcts_rl.logger import set_logger_level
from mcts_rl.utils import seed_everything, str2bool


def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='deepspeed --module mcts_rl.algorithms.ppo',
        description='Train language model using RLHF with PPO algorithm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument(
        '--actor_model_name_or_path',
        type=str,
        help='Path to the model checkpoint or its name.',
        required=True,
    )
    model_parser.add_argument(
        '--actor_ref_model_name_or_path',
        type=str,
        help='Path to the model checkpoint or its name.',
        required=True,
    )
    model_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
    )
    model_parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=128,
        help='The maximum sequence length of the model.',
    )
    model_parser.add_argument(
        '--trust_remote_code',
        type=str2bool,
        default=False,
        help='Whether to trust the remote code.',
    )
    model_parser.add_argument(
        '--reward_model_name_or_path',
        type=str,
        help='Path to the model checkpoint or its name.',
        default='',
    )

    # Dataset
    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument(
        '--use_code',
        action='store_true',
        default=False,
    )
    dataset_parser.add_argument(
        '--use_mcq',
        action='store_true',
        default=False,
    )
    dataset_parser.add_argument(
        '--few_shot',
        action='store_true',
        default=False,
    )
    dataset_parser.add_argument(
        '--train_datasets',
        type=parse_dataset,
        nargs='+',
        metavar='DATASET[:PROPORTION[:PATH]]',
        help='Dataset name(s) registered in the raw dataset.',
        required=True,
    )
    dataset_parser.add_argument(
        '--ptx_datasets',
        type=parse_dataset,
        nargs='*',
        metavar='DATASET[:PROPORTION[:PATH]]',
        help='Dataset name(s) registered in the raw dataset.',
    )
    dataset_parser.add_argument(
        '--eval_datasets',
        type=parse_dataset,
        nargs='+',
        metavar='DATASET[:PROPORTION[:PATH]]',
        help='Dataset name(s) registered in the raw dataset.',
    )

    # Training
    training_parser = parser.add_argument_group('training')
    training_parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
    )
    training_parser.add_argument(
        '--model_type',
        type=str,
        default='mistral',
        choices=['mistral', 'llama3', 'llama2', 'gpt-j'],
    )
    training_parser.add_argument(
        '--save_mcts_data',
        action='store_true',
        default=False,
    )
    training_parser.add_argument(
        '--filter',
        action='store_true',
        default=False,
    )
    training_parser.add_argument(
        '--norm_prob',
        action='store_true',
        default=False,
    )
    training_parser.add_argument(
        '--length_penalty',
        type=float,
        default=1.0,
    )
    training_parser.add_argument(
        '--ipo',
        action='store_true',
        default=False,
    )
    training_parser.add_argument(
        '--conservative',
        action='store_true',
        default=False,
    )
    training_parser.add_argument(
        '--choose_worst',
        action='store_true',
        default=False,
    )
    training_parser.add_argument(
        '--choose_random',
        action='store_true',
        default=True,
    )
    training_parser.add_argument(
        '--offline',
        action='store_true',
        default=False,
    )
    training_parser.add_argument(
        '--resume_from_ckpt',
        type=str,
        help='The directory to load optimizer and lr-scheduler.',
        default=None,
    )
    training_parser.add_argument(
        '--scale_coeff',
        type=float,
        default=0.02,
        help='The coefficient for the KL divergence between the reference and actor policy.',
    )
    training_parser.add_argument(
        '--kl_coeff',
        type=float,
        default=0.02,
        help='The coefficient for the KL divergence between the reference and actor policy.',
    )
    training_parser.add_argument(
        '--clip_range_ratio',
        type=float,
        default=0.2,
        help=(
            'The clipping range for ratio between the old and new policy. '
            'This is the epsilon parameter in the PPO algorithm.'
        ),
    )
    training_parser.add_argument(
        '--clip_range_score',
        type=float,
        default=50.0,
        help=(
            'The clipping range for the output of the score model. '
            'The reward is clipped into [-clip_range_score, clip_range_score].'
        ),
    )
    training_parser.add_argument(
        '--clip_range_value',
        type=float,
        default=5.0,
        help=(
            'The clipping range for the value function. '
            'The value is clipped into '
            '[value_estimate - clip_range_value, value_estimate + clip_range_value] '
            'during training.'
        ),
    )
    training_parser.add_argument(
        '--ptx_coeff',
        type=float,
        default=0.0,
        help='The coefficient for the ptx loss.',
    )
    training_parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Total number of training epochs to perform.',
    )
    training_parser.add_argument(
        '--update_iters',
        type=int,
        default=1,
        help='The number of repeated updates on a generated batch.',
    )
    training_parser.add_argument(
        '--per_device_prompt_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the training dataloader.',
    )
    training_parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the training dataloader.',
    )
    training_parser.add_argument(
        '--per_device_ptx_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the ptx training dataloader.',
    )
    training_parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the evaluation dataloader.',
    )
    training_parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of updates steps to accumulate before performing a backward/update pass.',
    )
    training_parser.add_argument(
        '--actor_lr',
        '--actor_learning_rate',
        type=float,
        default=1e-5,
        help='Initial learning rate (after the potential warmup period) for the actor model training.',
    )
    training_parser.add_argument(
        '--actor_weight_decay',
        type=float,
        default=0.0,
        help='Weight decay to for the actor model training.',
    )
    training_parser.add_argument(
        '--actor_lr_scheduler_type',
        type=SchedulerType,
        default='cosine',
        help='The scheduler type for actor model.',
        choices=[
            'linear',
            'cosine',
            'cosine_with_restarts',
            'polynomial',
            'constant',
            'constant_with_warmup',
        ],
    )
    training_parser.add_argument(
        '--actor_lr_warmup_ratio',
        type=float,
        default=0.0,
        help='Ratio of warm steps over total training steps for the actor lr scheduler.',
    )
    training_parser.add_argument(
        '--actor_gradient_checkpointing',
        action='store_true',
        help='Enable gradient checkpointing for actor model.',
    )
    training_parser.add_argument(
        '--normalize_reward',
        type=str2bool,
        default=False,
        help='Whether to normalize the reward during RL training.',
    )
    training_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='A seed for reproducible training.',
    )
    training_parser.add_argument(
        '--fp16',
        type=str2bool,
        default=False,
        help='Whether to use float16 precision.',
    )
    training_parser.add_argument(
        '--bf16',
        type=str2bool,
        default=False,
        help='Whether to use bfloat16 precision.',
    )
    training_parser.add_argument(
        '--tf32',
        type=str2bool,
        default=None,
        help='Whether to use tf32 mix precision.',
    )
    training_parser.add_argument(
        '--iteration_interval',
        type=int,
        default=32,
    )

    # Generation Config
    generation_parser = parser.add_argument_group('generation')
    generation_parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='The value used to module the next token probabilities.',
    )
    generation_parser.add_argument(
        '--init_temperature',
        type=float,
        default=1.0,
    )
    generation_parser.add_argument(
        '--top_p',
        type=float,
        default=1.0,
        help=(
            'If set to float < 1, only the smallest set of most probable tokens with '
            'probabilities that add up to`top_p` or higher are kept for generation.'
        ),
    )
    generation_parser.add_argument(
        '--num_return_sequences',
        type=int,
        default=1,
        help='The number of independently computed returned sequences for each element in the batch.',
    )
    generation_parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.',
    )
    
    # MCTS
    mcts_parser = parser.add_argument_group('mcts')
    mcts_parser.add_argument(
        '--eval_mode',
        action='store_true',
        default=False,
    )
    mcts_parser.add_argument(
        '--post',
        action='store_true',
        default=False,
    )
    mcts_parser.add_argument(
        '--reverse',
        action='store_true',
        default=False,
    )
    mcts_parser.add_argument(
        '--not_include_gt',
        action='store_true',
        default=False,
    )
    mcts_parser.add_argument(
        '--n_iters',
        type=int,
        default=5,
    )
    mcts_parser.add_argument(
        '--n_init_actions',
        type=int,
        default=2,
    )
    mcts_parser.add_argument(
        '--n_actions',
        type=int,
        default=2,
    )
    mcts_parser.add_argument(
        '--depth_limit',
        type=int,
        default=3,
    )
    mcts_parser.add_argument(
        '--breadth_limit',
        type=int,
        default=4,
    )
    mcts_parser.add_argument(
        '--force_terminating_on_depth_limit',
        action='store_true',
        default=False,
    )
    mcts_parser.add_argument(
        '--mcts_temperature',
        type=float,
        default=0.0,
    )
    mcts_parser.add_argument(
        '--mcts_temperature_decay_ratio',
        type=float,
        default=1.,
    )
    mcts_parser.add_argument(
        '--w_exp',
        type=float,
        default=1.,
    )
    mcts_parser.add_argument(
        '--no_self_eval',
        action='store_true',
        default=False,
    )
    mcts_parser.add_argument(
        '--mcts_length_penalty',
        type=float,
        default=1.25,
    )
    mcts_parser.add_argument(
        '--get_tp_zero',
        action='store_true',
        default=False,
    )

    # Evaluation
    evaluation_parser = parser.add_argument_group('evaluation')
    evaluation_parser.add_argument(
        '--eval_strategy',
        type=str,
        default='epoch',
        help='The evaluation strategy to adopt.',
        choices=['epoch', 'steps'],
    )
    evaluation_parser.add_argument(
        '--eval_interval',
        type=int,
        default=1000000,
        help='The interval to evaluate the model.',
    )
    evaluation_parser.add_argument(
        '--need_eval',
        default=False,
        help='Whether to evaluate the model during training.',
        action='store_true',
    )
    evaluation_parser.add_argument(
        '--eval_split_ratio',
        type=float,
        default=None,
        help='The split ratio of the evaluation dataset.',
    )
    evaluation_parser.add_argument(
        '--no_consider_diversity',
        default=False,
        action='store_true',
    )

    # Logging
    logging_parser = parser.add_argument_group('logging')
    logging_parser.add_argument(
        '--prediction_file_path',
        type=str,
        default=None,
        help='Where to store the model.',
    )
    logging_parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Where to store the model.',
    )
    logging_parser.add_argument(
        '--log_type',
        type=str,
        help='The type of logging.',
        default='wandb',
        choices=['wandb', 'tensorboard'],
    )
    logging_parser.add_argument(
        '--log_dir',
        type=str,
        help='The directory to store the logs.',
        default=None,
    )
    logging_parser.add_argument(
        '--log_project',
        type=str,
        help='The project name for logging.',
        default=None,
    )
    logging_parser.add_argument(
        '--log_run_name',
        type=str,
        help='The run name for logging.',
        default=None,
    )
    logging_parser.add_argument(
        '--save_16bit',
        action='store_true',
        help='Whether to save the model in 16-bit precision.',
    )
    logging_parser.add_argument(
        '--save_interval',
        type=int,
        default=1000000,
        help='The interval to save the model.',
    )

    # DeepSpeed
    deepspeed_parser = parser.add_argument_group('deepspeed')
    deepspeed_parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training on GPUs',
    )
    deepspeed_parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='ZeRO optimization stage for models.',
    )
    deepspeed_parser.add_argument(
        '--offload',
        type=str,
        default='none',
        choices=['none', 'parameter', 'optimizer', 'all'],
        help='Offload parameters and/or optimizer states to CPU.',
    )
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    if args.local_rank == -1:
        parser.error('`local_rank` not set, please use DeepSpeed launcher to run this script.')
    if args.fp16 and args.bf16:
        parser.error('Cannot use both bf16 and fp16 precision.')
    if args.bf16 and not is_torch_bf16_gpu_available():
        parser.error(
            'bf16 precision is not supported on this GPU. '
            'Please disable `--bf16` flag or use another precision flag (e.g., `--fp16`).',
        )
    if args.tf32 is not None and is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = args.tf32

    return args


def main() -> None:
    """Main training routine."""
    args = parse_arguments()

    deepspeed.init_distributed()

    args.global_rank = dist.get_rank()
    args.device = torch.device('cuda', args.local_rank)
    torch.cuda.set_device(args.device)
    seed_everything(args.seed)
    set_logger_level()

    dist.barrier()

    ds_train_config = get_deepspeed_train_config(
        micro_batch_size_per_gpu=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        stage=args.zero_stage,
        offload=args.offload,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    ds_eval_config = get_deepspeed_eval_config(
        stage=args.zero_stage,
        offload=args.offload,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    trainer = MCTSTrainer(args, ds_train_config, ds_eval_config)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    main()
