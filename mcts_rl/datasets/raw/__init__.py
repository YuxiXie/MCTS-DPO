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
"""Raw datasets."""

from mcts_rl.datasets.raw.alpaca import AlpacaDataset
from mcts_rl.datasets.raw.firefly import FireflyDataset
from mcts_rl.datasets.raw.hh_rlhf import (
    HhRLHFDialogueDataset,
    HhRLHFHarmlessDialogueDataset,
    HhRLHFHelpfulDialogueDataset,
)
from mcts_rl.datasets.raw.moss import MOSS002SFT, MOSS003SFT
from mcts_rl.datasets.raw.safe_rlhf import (
    SafeRLHF10KTrainDataset,
    SafeRLHFDataset,
    SafeRLHFTestDataset,
    SafeRLHFTrainDataset,
)
from mcts_rl.datasets.raw.prm800k import (
    PRM800KDataset,
    PRM800KTestDataset,
    PRM800KTrainDataset,
)
from mcts_rl.datasets.raw.mcq import (
    MCQDataset,
    SQATestDataset,
    SQATrainDataset,
    CSRTestDataset,
    CSRTrainDataset,
)
from mcts_rl.datasets.raw.math import (
    MATHDataset,
    MATHTestDataset,
    MATHTrainDataset,
)
from mcts_rl.datasets.raw.gsm8k import (
    GSM8KDataset,
    GSM8KTestDataset,
    GSM8KTrainDataset,
)
from mcts_rl.datasets.raw.arithmo import (
    ArithmoDataset,
    ArithmoTestDataset,
    ArithmoTrainDataset,
)
from mcts_rl.datasets.raw.qa_feedback import (
    QAFBDataset,
    QAFBTestDataset,
    QAFBTrainDataset,
)


__all__ = [
    'AlpacaDataset',
    'FireflyDataset',
    'HhRLHFDialogueDataset',
    'HhRLHFHarmlessDialogueDataset',
    'HhRLHFHelpfulDialogueDataset',
    'MOSS002SFT',
    'MOSS003SFT',
    'SafeRLHFDataset',
    'SafeRLHFTrainDataset',
    'SafeRLHFTestDataset',
    'SafeRLHF10KTrainDataset',
    'PRM800KDataset',
    'PRM800KTrainDataset',
    'PRM800KTestDataset',
    'MCQDataset',
    'SQATestDataset',
    'SQATrainDataset',
    'CSRTestDataset',
    'CSRTrainDataset',
    'MATHDataset',
    'MATHTrainDataset',
    'MATHTestDataset',
    'GSM8KDataset',
    'GSM8KTestDataset',
    'GSM8KTrainDataset',
    'ArithmoDataset',
    'ArithmoTestDataset',
    'ArithmoTrainDataset',
    'QAFBDataset',
    'QAFBTestDataset',
    'QAFBTrainDataset',
]
