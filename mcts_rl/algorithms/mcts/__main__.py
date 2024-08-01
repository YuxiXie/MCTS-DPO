"""The main training script to train RLHF using PPO algorithm."""

import sys

from mcts_rl.algorithms.mcts.main import main


if __name__ == '__main__':
    sys.exit(main())
