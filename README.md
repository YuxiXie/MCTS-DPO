# Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning

This repository contains code and analysis for the paper: [Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning](https://arxiv.org/abs/2405.00451). 
Below is the framework of our proposed method (on the left) together with a prompting example of self-evaluation (on the right).

![Model Framework](framework-colorblindfriendly.jpg)

#### Environment Setup

```sh
conda env create --file conda-recipe.yaml
pip install -r requirements.txt
```

#### Run MCTS-DPO

Our main code include `./mcts_rl/algorithms/mcts` and `./mcts_rl/trainers/tsrl_trainer.py`

To run MCTS-DPO for MathQA on Mistral (SFT):
```sh
bash scripts/mcts_mathqa.sh
```

To run MCTS-DPO for CSR on Mistral (SFT):
```sh
bash scripts/mcts_csr.sh
```

## Citation

```
@article{xie2024monte,
  title={Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning},
  author={Xie, Yuxi and Goyal, Anirudh and Zheng, Wenyue and Kan, Min-Yen and Lillicrap, Timothy P and Kawaguchi, Kenji and Shieh, Michael},
  journal={arXiv preprint arXiv:2405.00451},
  year={2024}
}
```

---
<sub><sup>This repository is adapted from the code of the works [Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf). </sup></sub>
