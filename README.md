# LLM-RL Financial Fraud Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements a hybrid fraud-detection pipeline that couples Large Language Model (LLM) embeddings with Reinforcement Learning (RL) agents to produce cost-sensitive, adaptive transaction screening policies. The implementation and experiments correspond to the paper "LLM-Assisted Fraud Detection with Reinforcement Learning" (PDF and LaTeX source under `paper/`). Read online : https://www.mdpi.com/1999-4893/18/12/792

**Quick summary**
- **Approach:** Use an LLM (e.g., DistilBERT / FinBERT) to encode transaction text + structured fields into embeddings; feed concatenated embeddings into an RL agent trained with an asymmetric reward that heavily penalizes missed frauds.
- **Key finding:** Policy-gradient agents (A2C, PPO) trained on LLM-derived states achieve substantially higher recall and better cost-sensitive utility than value-based (DQN) and contextual bandit baselines; A2C shows the strongest recall improvements on PaySim and competitive performance on the European Credit Card dataset.

**Representative figures (paper/images)**

- Workflow diagram (PDF): [paper/.../images/workflow.pdf](paper/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/images/workflow.pdf)

- Training / evaluation plots:

  ![DQN mean episode reward](paper/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/images/mean_ep_reward.png)

  ![DQN cumulative reward (eval)](paper/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/images/cumulative_rew_dqn.png)

  ![A2C policy loss](paper/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/images/policy_loss_a2c.png)

  ![A2C policy entropy](paper/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/images/policy_ent_a2c.png)

- Confusion matrices:

  ![DQN confusion matrix](paper/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/images/confusion_dqn.png)

  ![A2C confusion matrix (credit card)](paper/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/images/a2c_conf.png)

  ![A2C confusion matrix (PaySim)](paper/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/LLM-Assisted%20Fraud%20Detection%20with%20Reinforcement%20Learning/images/a2c_conf_pay.png)

Repository layout (essential parts):

```
paper/    # Full manuscript (PDF + LaTeX) + figures
notebooks/ # Notebooks used to run experiments and reproduce results
src/       # Environment and helper modules (e.g., src/custom_env.py)
data/      # (large datasets/embeddings kept out of repo)
results/   # Generated plots, checkpoints, and tensorboard logs
```

**Findings & Recommendations**
- **LLM+RL synergy:** Combining semantic text embeddings with structured features yields richer states and improves recall under class imbalance.
- **Algorithm choice:** Actor–critic / policy-gradient (A2C, PPO) perform best for high-recall, cost-sensitive objectives; DQN is competitive on balanced performance but struggles with rare-event sampling.
- **Reward design:** Asymmetric rewards (large negative for false negatives) are essential to align the agent with business costs.
- **Deployment note:** The approach is adaptive and suitable for online updates; careful monitoring and human-in-the-loop review are recommended for production use.

How to reproduce (quick)
- Install dependencies:
```bash
pip install -r requirements.txt
```
- Open and run the main notebook(s):
  - `notebooks/RL2.0_ATT.ipynb` (contains data prep, embedding generation, RL training)
- Visualize training with TensorBoard:
```bash
tensorboard --logdir=results/tensorboard/
```

Where to look in this repo
- **Paper:** `paper/LLM-Assisted Fraud Detection with Reinforcement Learning/Fraud_detection.pdf` and `Fraud_detection.tex` (LaTeX source and bibliography).
- **Figures:** `paper/.../images/` (contains all PNGs/PDFs referenced in the manuscript).
- **Environment:** `src/custom_env.py` — Gym-compatible environment implementing the reward scheme and observation (LLM embedding + structured features).
- **Notebooks:** `notebooks/` — runnable experiments and utility notebooks for embedding generation and model training.

Cite / read online
- Read the full paper online: https://www.mdpi.com/1999-4893/18/12/792

License
- MIT — see LICENSE file.

Contact
- Open an issue or contact the repository owners for questions or reproduction help.
