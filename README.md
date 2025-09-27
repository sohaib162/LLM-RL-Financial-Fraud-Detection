# RL for Transaction Fraud Detection (PaySim & Credit Card)

> Reproducible notebooks for reinforcement learning (A2C/DQN/PPO) applied to transaction fraud detection, with classic ML/NLP baselines and portable data loaders.

## Overview
This repository collects our Final Year Project (FYP) experiments on **reinforcement learning (RL)** for transaction fraud detection. We provide notebooks that:
- Build **custom Gymnasium environments** over transaction **embeddings** (e.g., 768‑dim sentence-transformers).
- Train **A2C, DQN, and PPO** (Stable-Baselines3) to learn fraud‑aware policies and evaluation strategies.
- Include **supervised baselines** (e.g., DistilBERT/embeddings + classifier) for comparison.
- Target two public datasets: **PaySim** and **Credit Card Fraud (Kaggle/MLG‑ULB)**.

The goal is to make experiments **reproducible** on a standard Ubuntu laptop (CPU or single NVIDIA GPU), avoiding Colab/Kaggle‑specific paths. A short report summarizing the approach and findings can be added to `docs/` when available.

## Highlights
- **Custom RL environment** wrapping transaction embeddings (default 768‑d) with shaped rewards.
- **Stable-Baselines3** pipelines for A2C, DQN, PPO, with VecEnv/VecNormalize options.
- **Portable data loaders** powered by `kagglehub` with local fallbacks—no absolute `/content` or `/kaggle/input` paths.
- **Supervised baselines** (DistilBERT embeddings + classifier) to contextualize RL performance.
- Clear **repro steps** and pinned **requirements**.

## Project Structure
```
.
├─ data/                          # Auto-downloaded or manual CSVs (git-ignored)
├─ notebooks/
│  ├─ paysim/
│  │  ├─ RL_PAYSIM.ipynb
│  │  └─ paysim_processing.ipynb  # (renamed from paysim_processingipynb)
│  ├─ credit_card/
│  │  ├─ RL2.0.ipynb
│  │  └─ distilbert_baseline.ipynb
│  └─ fillings/
│     └─ RL_FILLINGS.ipynb        # expects Final_Dataset.csv
├─ requirements.txt
└─ README.md
```

If you start from an existing tree, rename:
- `cards fraud_detection/` → `notebooks/credit_card/`
- `paysim/paysim_processingipynb` → `notebooks/paysim/paysim_processing.ipynb`
- `fraud-detection-with-distilber_finalt.ipynb` → `distilbert_baseline.ipynb`

## Datasets
- **PaySim** (Kaggle): `ealaxi/paysim1` → file: `PS_20174392719_1491204439457_log.csv`
- **Credit Card Fraud** (Kaggle/MLG‑ULB): `mlg-ulb/creditcardfraud` → file: `creditcard.csv`
- **Fillings** notebook expects `Final_Dataset.csv` (place in `data/` or adapt loader).

### Portable data loading (drop-in cells)
**Credit Card Fraud**
```python
import os, pandas as pd
try:
    import kagglehub
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df = pd.read_csv(os.path.join(path, "creditcard.csv"))
except Exception:
    df = pd.read_csv("data/creditcard.csv")  # manual fallback
```

**PaySim**
```python
import os, pandas as pd
try:
    import kagglehub
    path = kagglehub.dataset_download("ealaxi/paysim1")
    df = pd.read_csv(os.path.join(path, "PS_20174392719_1491204439457_log.csv"))
except Exception:
    df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")
```

## Setup
Python ≥ 3.10 is recommended.

```bash
# Clone
git clone https://github.com/<you>/rl-transaction-fraud.git
cd rl-transaction-fraud

# Create env (conda example)
conda create -n rl-fraud python=3.10 -y
conda activate rl-fraud

# Install deps
python -m pip install -r requirements.txt
```

### requirements.txt
We pin core tools for Gymnasium + SB3 and standard ML / NLP stacks.
```
# RL
stable-baselines3==2.3.2
gymnasium==0.29.1

# Core scientific
numpy>=1.26
pandas>=2.2
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13

# NLP / embeddings
transformers>=4.42
sentence-transformers>=3.0
datasets>=2.20
torch>=2.3

# Data helpers
kagglehub>=0.2.5

# Optional
ctgan>=0.7
plotly>=5.22
```

## Quickstart
1. Open the notebook you want (e.g., `notebooks/paysim/RL_PAYSIM.ipynb`).  
2. Run the **dataset loader** cell (above) to fetch or read the CSV.  
3. Ensure your environment imports **Gymnasium** (not legacy `gym`):
   ```python
   import gymnasium as gym
   from gymnasium import spaces
   ```
4. Train an RL agent (A2C/DQN/PPO) and monitor training logs/plots.  
5. Save artifacts to `artifacts/`:
   ```python
   model.save("artifacts/a2c_paysim.zip")
   ```

## Custom RL Environment
The RL notebooks define a `FraudDetectionEnv(gym.Env)` variant operating on **embedding vectors** (default 768‑dim). If you abstract it into a module, place code in `envs/fraud_env.py` and import it in notebooks. Validate the observed feature dimension and surface helpful errors (e.g., `print(obs.shape)` when mismatched).

## Baselines
- **DistilBERT embeddings + classic classifier** notebook lives under `notebooks/credit_card/distilbert_baseline.ipynb`.
- Consider reporting precision/recall, ROC‑AUC, PR‑AUC, and calibration, alongside RL metrics.

## Reproducibility
- Clear heavy notebook outputs before committing:
  ```bash
  find notebooks -name "*.ipynb" -print0 | xargs -0 -I{}     jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {}
  ```
- Fix random seeds where possible (NumPy, PyTorch, SB3).  
- Prefer **VecNormalize** and document `n_envs` if using parallel envs.

## Results (to be filled)
Add your tables/figures comparing RL agents vs supervised baselines on PaySim and CreditCard datasets. Include training curves, confusion matrices, and cost‑sensitive metrics if relevant.

## Roadmap
- Package `FraudDetectionEnv` as a pip‑installable module.
- Add hyperparameter sweeps (Optuna) and PR‑AUC optimization.
- Explore reward shaping & cost‑aware policies (class imbalance).
- Add CI pre‑commit hooks for lint & notebook output clearing.


## Acknowledgements
- PaySim and MLG‑ULB Credit Card Fraud datasets (Kaggle).
- Stable‑Baselines3, Gymnasium, Hugging Face Transformers/Sentence‑Transformers.
