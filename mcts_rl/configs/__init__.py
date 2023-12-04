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
"""Configurations and constants."""

from mcts_rl.configs import constants
from mcts_rl.configs.constants import *  # noqa: F403
from mcts_rl.configs.deepspeed_config import (
    TEMPLATE_DIR,
    get_deepspeed_eval_config,
    get_deepspeed_train_config,
)


__all__ = [
    *constants.__all__,
    'TEMPLATE_DIR',
    'get_deepspeed_eval_config',
    'get_deepspeed_train_config',
]
