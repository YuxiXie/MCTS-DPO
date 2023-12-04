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
"""Trainer class for supervised finetuning."""

from __future__ import annotations

from typing import Any
from tqdm import tqdm

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from mcts_rl.datasets import SupervisedDataset
from mcts_rl.trainers import SupervisedTrainer
from mcts_rl.utils import (
    get_all_reduce_mean, 
    is_main_process, 
    to_device,
)


class SupervisedFinetuneTrainer(SupervisedTrainer):
    """Trainer class for supervised finetuning."""

    TRAINING_TYPE = 'sft'
    DATASET_TYPE = SupervisedDataset
    MODEL_TYPE = AutoModelForCausalLM

    def loss(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            'loss': outputs.loss,
        }
    
    def eval(
        self,
    ) -> dict[str, Any]:
        if self.eval_dataloader is None:
            return {}
        
        self.set_eval()
        eval_dataloader = tqdm(
            self.eval_dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
            position=1,
            leave=False,
        )

        losses, batch = [], None
        for batch in eval_dataloader:
            batch = to_device(batch, self.args.device)
            with torch.no_grad():
                loss = self.loss(
                    input_ids=batch['input_ids'],
                    labels=batch['labels'],
                    attention_mask=batch['attention_mask'],
                )['loss']
            losses.extend([loss])
        
        if batch is None:
            self.logger.print('WARNING: `eval_dataloader` is empty.')
            return {}
        
        losses = torch.stack(losses, dim=0)
        if is_main_process():
            gathered_losses = [torch.empty_like(losses) for _ in range(dist.get_world_size())]
        else:
            gathered_losses = []
        dist.gather(losses, gathered_losses, dst=0)
        if is_main_process():
            losses = torch.cat(gathered_losses, dim=0)
        
        self.set_train()
        
        return {
            'eval/loss': losses.mean().item(),
        }

    def train_step(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
    ) -> dict[str, Any]:
        """Performs a single training step.

        Args:
            input_ids (torch.LongTensor): input ids for causal inputs to complete with.
            labels (torch.LongTensor): labels for the full sequence.
            attention_mask (torch.BoolTensor): attention mask for the labels.

        Returns:
            dict[str, Any]: training loss, learning rate
        """
        loss = self.loss(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )['loss']
        self.model.backward(loss)
        self.model.step()

        loss = get_all_reduce_mean(loss)

        return {
            'train/loss': loss.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }
