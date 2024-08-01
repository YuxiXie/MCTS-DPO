"""Trainer base class for RL training."""

from __future__ import annotations

import os
import abc
import optree
import argparse
import copy
import itertools
import jsonlines
from tqdm import tqdm
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from accelerate import skip_first_batches
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedModel,
    get_scheduler,
)
from transformers.deepspeed import HfDeepSpeedConfig, deepspeed_load_checkpoint

from mcts_rl.configs import ADAM_BETAS
from mcts_rl.datasets import (
    SupervisedDataset,
    DummyDataset,
    PromptOnlyBatch, PromptOnlyDataset,
    PromptOnlyPostBatch, PromptOnlyPostDataset,
)
from mcts_rl.models import AutoModelForScore, load_pretrained_models
from mcts_rl.trainers.base import TrainerBase
from mcts_rl.utils import (
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    is_main_process,
    to_device,
    check_available,
)


class TSRLTrainer(TrainerBase):  # pylint: disable=too-many-instance-attributes
    """Trainer base class for Tree Search RL training.

    Abstract methods:
        rollout: Rollout a batch of experiences.
        rl_step: Perform a single update step with RL loss.
        eval_step: Perform a single evaluation step.
    """

    TRAINING_TYPE: ClassVar[str] = 'tsrl'

    actor_model: deepspeed.DeepSpeedEngine
    actor_reference_model: deepspeed.DeepSpeedEngine

    ds_train_config: dict[str, Any]
    ds_eval_config: dict[str, Any]

    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_train_config = ds_train_config
        self.ds_eval_config = ds_eval_config
        self.global_step = 0

        self.init_models()
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        
        self.generation_config = GenerationConfig(
            max_length=self.args.max_length,
            max_new_tokens=self.args.max_new_tokens,
            num_return_sequences=self.args.num_return_sequences,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
            do_sample=True,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.init_mcts_searcher()
        dist.barrier()
        self.init_logger()

        # Those value can be changed
        self.kl_coeff = self.args.kl_coeff
        self.clip_range_ratio = self.args.clip_range_ratio
        self.clip_range_score = self.args.clip_range_score
        self.clip_range_value = self.args.clip_range_value
        self.ptx_coeff = self.args.ptx_coeff
        self.gamma = 1.0
        self.gae_lambda = 0.95
        self.scale_coeff = self.args.scale_coeff

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if (
            self.ds_train_config is not None
            and self.ds_train_config['zero_optimization']['stage'] == 3
        ):
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_config)

        if (
            self.ds_eval_config is not None
            and self.ds_eval_config['zero_optimization']['stage'] == 3
        ):
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_config)

        self.actor_model, self.tokenizer = load_pretrained_models(
            self.args.actor_model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )
        self.actor_reference_model, _ = load_pretrained_models(
            self.args.actor_ref_model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )
        
        self.use_reward_model = False
        if self.args.reward_model_name_or_path:
            self.use_reward_model = True
            self.reward_model, self.reward_tokenizer = load_pretrained_models(
                self.args.reward_model_name_or_path,
                model_max_length=self.args.max_length,
                auto_model_type=AutoModelForScore,
                padding_side='right',
                trust_remote_code=self.args.trust_remote_code,
                auto_model_kwargs={
                    'score_type': 'reward',
                    'do_normalize': self.args.normalize_reward,
                },
            )
            self.reward_model.set_normalize(self.args.normalize_reward)

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        if (
            self.args.per_device_prompt_batch_size
            * self.args.num_return_sequences
            % self.args.per_device_train_batch_size
            != 0
        ):
            raise ValueError(
                'The number of prompt-only samples must be divisible by the micro batch size.',
            )

        prompt_only_dataset = PromptOnlyDataset(
            self.args.train_datasets,
            tokenizer=self.tokenizer,
            use_mcq=self.args.use_mcq,
            few_shot=self.args.few_shot,
            model_type=self.args.model_type,
        ) if not self.args.post else PromptOnlyPostDataset(
            self.args.train_datasets,
            tokenizer=self.tokenizer,
        )

        if self.args.need_eval:
            if self.args.eval_datasets is None and self.args.eval_split_ratio is not None:
                prompt_only_dataset, eval_dataset = prompt_only_dataset.split_train_test(
                    split_ratio=self.args.eval_split_ratio,
                )
                self.eval_dataloader = DataLoader(
                    eval_dataset,
                    collate_fn=eval_dataset.get_collator(),
                    sampler=DistributedSampler(eval_dataset, shuffle=True),
                    batch_size=self.args.per_device_eval_batch_size,
                )
            elif self.args.eval_datasets is not None and self.args.eval_split_ratio is None:
                eval_dataset = PromptOnlyDataset(
                    self.args.eval_datasets,
                    tokenizer=self.tokenizer,
                    use_mcq=self.args.use_mcq,
                    few_shot=self.args.few_shot,
                    model_type=self.args.model_type,
                )
                self.eval_dataloader = DataLoader(
                    eval_dataset,
                    collate_fn=eval_dataset.get_collator(),
                    sampler=DistributedSampler(eval_dataset, shuffle=True),
                    batch_size=self.args.per_device_eval_batch_size,
                )
            else:
                raise ValueError('Either `eval_datasets` or `eval_split_ratio` should be provided.')
        else:
            self.eval_dataloader = None

        self.prompt_only_dataloader = DataLoader(
            prompt_only_dataset,
            collate_fn=prompt_only_dataset.get_collator(),
            sampler=DistributedSampler(prompt_only_dataset, shuffle=True),
            batch_size=self.args.per_device_prompt_batch_size,
        )
        
        self.use_ptx = self.args.ptx_datasets is not None
        if self.use_ptx:
            ptx_dataset = SupervisedDataset(
                self.args.ptx_datasets,
                tokenizer=self.tokenizer,
            )

            self.ptx_dataloader = DataLoader(
                ptx_dataset,
                collate_fn=ptx_dataset.get_collator(),
                sampler=DistributedSampler(ptx_dataset, shuffle=True),
                batch_size=self.args.per_device_ptx_batch_size,
            )
        else:
            self.ptx_dataloader = DataLoader(DummyDataset(len(self.prompt_only_dataloader)))

        self.args.total_training_steps = int(
            len(self.prompt_only_dataloader)
            * self.args.epochs
            * self.args.update_iters
            * self.args.per_device_prompt_batch_size
            * self.args.num_return_sequences
            // self.args.per_device_train_batch_size,
        )

    def _init_train_engine(
        self,
        model: nn.Module,
        weight_decay: float,
        lr: float,
        lr_scheduler_type: str,
        lr_warmup_ratio: float,
        total_training_steps: int,
        ds_config: dict[str, Any],
    ) -> deepspeed.DeepSpeedEngine:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, weight_decay)
        if (
            ds_config['zero_optimization'].get('offload_optimizer', {}).get('device', 'none')
            != 'none'
        ):
            optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=lr, betas=ADAM_BETAS)
        else:
            optimizer = FusedAdam(optimizer_grouped_parameters, lr=lr, betas=ADAM_BETAS)

        lr_scheduler_update_steps = total_training_steps // ds_config['gradient_accumulation_steps']
        num_warmup_steps = int(lr_scheduler_update_steps * lr_warmup_ratio)
        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=lr_scheduler_update_steps,
        )
        engine, *_ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=ds_config,
        )
        if self.args.resume_from_ckpt is not None:
            deepspeed_load_checkpoint(engine, self.args.resume_from_ckpt)
        return engine

    def _init_eval_engine(
        self,
        model: nn.Module,
        ds_config: dict[str, Any],
    ) -> deepspeed.DeepSpeedEngine:
        engine, *_ = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
        return engine

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        actor_ds_config = copy.deepcopy(self.ds_train_config)
        actor_total_training_steps = self.args.total_training_steps
        if self.use_ptx:
            self.args.gradient_accumulation_steps *= 2
            actor_ds_config['train_batch_size'] *= 2
            actor_ds_config['gradient_accumulation_steps'] *= 2
            actor_total_training_steps *= 2
        
        if self.args.need_eval:
            self.actor_model = self._init_eval_engine(
                model=self.actor_model,
                ds_config=self.ds_eval_config,
            )
            self.actor_model.eval()
        else:
            self.actor_model = self._init_train_engine(
                model=self.actor_model,
                weight_decay=self.args.actor_weight_decay,
                lr=self.args.actor_lr,
                lr_scheduler_type=self.args.actor_lr_scheduler_type,
                lr_warmup_ratio=self.args.actor_lr_warmup_ratio,
                total_training_steps=actor_total_training_steps,
                ds_config=actor_ds_config,
            )

        self.actor_reference_model = self._init_eval_engine(
            model=self.actor_reference_model,
            ds_config=self.ds_eval_config,
        )
        self.actor_reference_model.eval()
        
        if self.use_reward_model:
            self.reward_model = self._init_eval_engine(
                model=self.reward_model,
                ds_config=self.ds_eval_config,
            )
            self.reward_model.eval()

    @abc.abstractmethod
    def init_mcts_searcher(self) -> None:
        raise NotImplementedError
    
    def set_train(self, mode: bool = True) -> None:
        """Set training mode for all models."""
        if mode:
            self.actor_model.train()

            if self.args.actor_gradient_checkpointing:
                self.actor_model.gradient_checkpointing_enable()
        else:
            self.actor_model.eval()

            if self.args.actor_gradient_checkpointing:
                self.actor_model.gradient_checkpointing_disable()

    def split_tsrl_micro_batches(
        self,
        prompt_only_batch: PromptOnlyBatch | PromptOnlyPostBatch,
    ) -> list[PromptOnlyBatch | PromptOnlyPostBatch]:
        """Split a batch of RL samples into micro-batches."""
        total_batch_size = prompt_only_batch['input_ids'].size(0) if not self.args.post \
            else len(prompt_only_batch['prompts_list'])
        micro_batch_size = self.args.per_device_train_batch_size
        micro_batches = []
        assert total_batch_size == micro_batch_size
        micro_batches.extend(self.tree_constructor(prompt_only_batch))
        return micro_batches

    def split_ptx_micro_batches(
        self,
        ptx_batch: dict[str, torch.Tensor],
    ) -> list[dict[str, torch.Tensor]]:
        """Split a batch of PTX samples into micro-batches."""
        micro_batches = []
        total_batch_size = ptx_batch['input_ids'].size(0)
        micro_batch_size = self.args.per_device_ptx_batch_size
        for i in range(0, total_batch_size, micro_batch_size):
            micro_batch = optree.tree_map(
                # pylint: disable-next=cell-var-from-loop
                lambda tensor: tensor[i : i + micro_batch_size],  # noqa: B023
                ptx_batch,
            )
            micro_batches.append(micro_batch)
        return micro_batches
    
    @abc.abstractmethod
    @torch.no_grad()
    def tree_constructor(self, prompt_only_batch: PromptOnlyBatch | PromptOnlyPostBatch) -> list[dict[str, Any]]:
        """Rollout a batch of experiences."""
        raise NotImplementedError

    @abc.abstractmethod
    @torch.no_grad()
    def post_tree_construct(
        self,
        prompt: torch.Tensor,
        target_probs: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Post-process a rollout sample."""
        raise NotImplementedError

    @abc.abstractmethod
    def tsrl_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform a single update step with RL loss."""
        raise NotImplementedError

    def ptx_step(self, ptx_batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Perform a single update step with PTX loss."""
        torch.cuda.empty_cache()
        try:
            ptx_loss = self.actor_model(
                input_ids=ptx_batch['input_ids'],
                attention_mask=ptx_batch['attention_mask'],
                labels=ptx_batch['labels'],
            ).loss
            self.actor_model.backward(self.ptx_coeff * ptx_loss)
        except Exception as e:
            print('\n{}'.format(str(e)))
        self.actor_model.step()

        ptx_loss = get_all_reduce_mean(ptx_loss)

        return {
            'train/ptx_loss': ptx_loss.item(),
        }
    
    def train(self) -> None:
        """Train the model."""
        self.logger.print('***** Running training *****')        
        progress_bar = tqdm(
            total=self.args.total_training_steps,
            desc=f'Training 1/{self.args.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )
        
        steps_trained_in_current_epoch, epochs_trained = 0, 0
        if self.args.resume_from_ckpt is not None:
            if self.use_ptx:
                steps_trained_in_current_epoch = self.actor_model.global_steps * self.args.gradient_accumulation_steps // 2
            else:
                steps_trained_in_current_epoch = self.actor_model.global_steps * self.args.gradient_accumulation_steps
            if steps_trained_in_current_epoch > 0:
                progress_bar.update(steps_trained_in_current_epoch)
            self.global_step = steps_trained_in_current_epoch
            epochs_trained = steps_trained_in_current_epoch // len(self.prompt_only_dataloader)
            steps_trained_in_current_epoch %= len(self.prompt_only_dataloader)
            if not steps_trained_in_current_epoch:
                _step = int(self.args.resume_from_ckpt.split('/')[-1].replace('steps', ''))
                steps_trained_in_current_epoch = _step
                progress_bar.update(steps_trained_in_current_epoch)
                self.global_step = steps_trained_in_current_epoch
            steps_trained_in_current_epoch = int(steps_trained_in_current_epoch * 5/4)    # avoid duplication

        if self.args.need_eval and self.eval_dataloader is not None:
            self.logger.print('\n***** Evaluating at the beginning *****')
            self.logger.log(self.eval(), step=0)
        
        num_prompt_only_batches = len(self.prompt_only_dataloader)
        num_ptx_batches = len(self.ptx_dataloader)
        num_ptx_replicas = (num_prompt_only_batches + num_ptx_batches - 1) // num_ptx_batches
        
        for epoch in range(self.args.epochs):
            if epoch < epochs_trained: continue
            
            for prompt_only_batch, ptx_batch in zip(
                self.prompt_only_dataloader,
                itertools.chain.from_iterable([self.ptx_dataloader] * num_ptx_replicas),
            ):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                
                # generate batches
                if not self.args.offline:
                    self.set_eval()
                prompt_only_batch = to_device(prompt_only_batch, self.args.device)
                rl_batches = self.split_tsrl_micro_batches(prompt_only_batch)
                if self.use_ptx:
                    ptx_batch = to_device(ptx_batch, self.args.device)
                    ptx_batches = self.split_ptx_micro_batches(ptx_batch)
                else:
                    ptx_batches = [None for _ in range(len(rl_batches))]
                torch.cuda.empty_cache()
                
                # train
                self.set_train()
                for _ in range(self.args.update_iters):
                    for rl_batch, ptx_batch in zip(rl_batches, ptx_batches):
                        if not check_available(rl_batch, 
                                               eos_token_id=self.tokenizer.convert_tokens_to_ids("<|eot_id|>") if self.args.model_type == 'llama3' else self.tokenizer.eos_token_id,
                                               max_tokens=self.args.max_length, 
                                               to_filter=self.args.filter):
                            continue
                        rl_info = self.tsrl_step(**rl_batch)
                        if rl_info is None or not len(rl_info): continue
                        torch.cuda.empty_cache()
                        self.logger.log(rl_info, step=self.global_step)
                        if self.use_ptx:
                            ptx_info = self.ptx_step(ptx_batch)
                            torch.cuda.empty_cache()
                            self.logger.log(ptx_info, step=self.global_step)
                        
                        self.global_step += 1
                        progress_bar.set_description(
                            f'Training {epoch + 1}/{self.args.epochs} epoch '
                        )
                        progress_bar.update(1)
                        
                        if self.args.save_mcts_data:
                            prompt = self.tokenizer.batch_decode(rl_batch['prompts_list'][0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                            generated = [self.tokenizer.batch_decode(seq, skip_special_tokens=False, clean_up_tokenization_spaces=False) for seq in rl_batch['input_ids_list']]
                            init_values = [x for x in rl_batch['init_value_list']]
                            # generated = [[text[len(prompt[0]):] for text in text_list] for i, text_list in enumerate(generated)]
                            generated = [[text for text in text_list] for i, text_list in enumerate(generated)]
                            with jsonlines.open(os.path.join(self.args.output_dir, 'mcts_rst_data.jsonl'), mode='a') as writer:
                                writer.write_all([{
                                    'prompt': prompt[0], 
                                    'generated': generated,
                                    'init_values': init_values,
                                }])
                        
                        if self.global_step % self.args.save_interval == 0:
                            self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                            self.actor_model.save_checkpoint(
                                self.args.output_dir,
                                tag=self.global_step,
                            )
                            self.save(global_steps=self.global_step)
                            self.logger.print('Checkpoint saved.')

                        if (
                            self.args.need_eval
                            and self.args.eval_strategy == 'steps'
                            and self.global_step % self.args.eval_interval == 0
                        ):
                            self.logger.print(
                                f'\n***** Evaluating at step {self.global_step} *****',
                            )
                            self.logger.log(self.eval(), step=self.global_step)

            if self.args.need_eval: # and self.args.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.args.epochs} *****',
                )
                self.save(global_steps=self.global_step)
                self.logger.log(self.eval(), step=self.global_step)

    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            return {}

        self.set_eval()
        prompts: list[str] = []
        generateds: list[str] = []

        eval_dataloader = tqdm(
            self.eval_dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
        )
        
        outputfile = self.args.prediction_file_path
        count = 0
        if os.path.exists(outputfile):
            with jsonlines.open(outputfile, mode='r') as reader:
                count = len([x for x in reader])
        if '/scoring/' in outputfile:
            correct_token_ids = [self.tokenizer.encode(tok)[1] for tok in ['B', 'correct', 'Correct']]
            correct_token_ids += [self.tokenizer.encode(tok)[-1] for tok in ['(B', ' B', ' correct']]
            if len(self.tokenizer.encode('Correct')) < 3:
                correct_token_ids += [self.tokenizer.encode(tok)[-1] for tok in [' Correct']]
            correct_token_ids = list(set(correct_token_ids))
        
        idx = -1
        for batch in eval_dataloader:
            idx += 1
            if idx < count: continue
            if '/mj2/' in outputfile:
                if idx > 400:
                    break
            batch = to_device(batch, self.args.device)
            if '/mcts/' in outputfile:
                rl_batch = self.split_tsrl_micro_batches(batch)[0]
                torch.cuda.empty_cache()
            elif self.args.num_return_sequences > 1:
                with torch.no_grad():
                    seq = self.actor_model.module.generate(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        max_length=self.args.max_length,
                        synced_gpus=True,
                        do_sample=True,
                        num_return_sequences=self.args.num_return_sequences,
                        temperature=self.args.temperature,
                    )
            elif '/scoring/' in outputfile:
                with torch.no_grad():
                    sequences = self.actor_model.module.generate(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        max_new_tokens=8,
                        synced_gpus=True,
                        do_sample=False,
                        num_return_sequences=self.args.num_return_sequences,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                    seq, scores = sequences.sequences, sequences.scores
                    print(self.tokenizer.decode(seq[0]))
                    conf = 0.0
                    for idx, _id in enumerate(seq[0]):
                        if idx < batch['input_ids'].size(-1): continue
                        if self.tokenizer.decode(_id).strip() in ['A', 'B', 'correct', 'wrong', 'incorrect']:
                            logprobs = F.log_softmax(scores[idx - batch['input_ids'].size(-1)][0], dim=-1)
                            conf = sum(torch.exp(logprobs[tok_id]).detach().item() for tok_id in correct_token_ids)
                            break
                    if conf == 0:
                        for idx, _id in enumerate(seq[0]):
                            if idx < batch['input_ids'].size(-1): continue
                            if self.tokenizer.decode(_id).strip() in ['Cor', 'In', 'A', 'B', 'correct', 'wrong', 'incorrect']:
                                logprobs = F.log_softmax(scores[idx - batch['input_ids'].size(-1)][0], dim=-1)
                                conf = sum(torch.exp(logprobs[tok_id]).detach().item() for tok_id in correct_token_ids)
                                break
                    print(conf)
            else:
                terminators = [self.tokenizer.eos_token_id]
                terminators += [self.tokenizer.convert_tokens_to_ids("<|eot_id|>")] if self.args.model_type == 'llama3' else []
                
                if batch['input_ids'].size(-1) >= self.args.max_length:
                    prompts.append(self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True))
                    generateds.append('')
                    continue
                
                with torch.no_grad():
                    seq = self.actor_model.module.generate(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        max_length=self.args.max_length,
                        synced_gpus=True,
                        do_sample=False,
                        eos_token_id=terminators,
                    )

            dist.barrier()

            prompt = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            init_values = None
            if '/mcts/' in outputfile:
                generated = [self.tokenizer.batch_decode(seq, skip_special_tokens=True) for seq in rl_batch['input_ids_list']]
                init_values = [x for x in rl_batch['init_value_list']]
            else:
                generated = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
            if '/mcts/' in outputfile:
                generated = [[text[len(prompt[0]) :] for text in text_list] for i, text_list in enumerate(generated)]
            else:
                generated = [text[len(prompt[0]) :] for text in generated]
            
            prompts.extend(prompt)
            generateds.extend(generated)
            
            with jsonlines.open(outputfile, mode='a') as writer:
                writer.write_all([{
                    'prompt': prompt, 
                    'generated': generated,
                    'answer': batch['answer'][0],
                    'answer_content': batch['answer_content'][0],
                    'score': conf if '/scoring/' in outputfile else -1,
                    'init_values': init_values,
                }])

        dist.barrier()

        self.set_train()
        # import ipdb; ipdb.set_trace()
        assert False, """Cannot do eval & train in one process"""
        return {'num': len(generateds)}

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | PreTrainedModel | None = None,
        ds_config: dict | None = None,
        global_steps: int = -1,
    ) -> None:
        """Save model and tokenizer."""
        if model is None:
            model = self.actor_model
        if ds_config is None:
            ds_config = self.ds_train_config
        super().save(model=model, ds_config=ds_config, global_steps=global_steps)
