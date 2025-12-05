# LLM-RL Financial Fraud Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Multi-Agent Reinforcement Learning (MARL) system for financial fraud detection that leverages Large Language Model (LLM) embeddings and state-of-the-art RL algorithms.

## Overview

This project implements a novel approach to financial fraud detection by combining:
- **LLM Embeddings**: DistilBERT-generated transaction embeddings
- **Reinforcement Learning**: DQN and A2C agents for fraud classification
- **Data Augmentation**: CTGAN for synthetic fraud sample generation
- **Dimensionality Reduction**: PCA for efficient training

## Key Features

- **Hybrid LLM-RL Architecture**: Uses attention-pooled embeddings from DistilBERT as state representations
- **Multi-Algorithm Support**: Implements both DQN (Deep Q-Network) and A2C (Advantage Actor-Critic)
- **Class Imbalance Handling**: CTGAN-based synthetic fraud generation
- **Comprehensive Evaluation**: Confusion matrices, precision, recall, F1-score metrics
- **TensorBoard Integration**: Real-time training and evaluation visualization

## Performance

### DQN Agent
- **Accuracy**: 91.03%
- **Precision**: 66.42%
- **Recall**: 92.86%
- **F1-Score**: 77.45%

### A2C Agent
- **Accuracy: 99.%**
- **Precision: 100%**
- **Recall: 99.7%**
- **F1-Score: 99.9%**

## Project Structure

```
LLM-RL-Financial-Fraud-Detection/
├── notebooks/              # Jupyter notebooks for experiments
│   ├── RL2.0_ATT.ipynb    # Main RL training with attention embeddings
│   ├── RL2.0_ATT_Hybrid.ipynb
│   ├── RL2.0.ipynb
│   └── fraud-detection-with-distilbert.ipynb
├── src/                    # Source code
│   └── custom_env.py      # Custom Gymnasium environment
├── data/                   # Data files
│   └── embeddings/        # Pre-generated embeddings
│       ├── embeddings.pkl
│       └── attention_pooled_embeddings.pkl
├── results/               # Training results
│   ├── figures/          # Generated plots and visualizations
│   ├── checkpoints/      # Model checkpoints
│   └── tensorboard/      # TensorBoard logs
├── models/               # Saved trained models
│   ├── dqn_fraud_model.zip
│   └── a2c_fraud_model.zip
├── .models/              # Alternative model storage
├── ablation_study/       # Ablation study experiments
├── paper/                # Research paper and documentation
└── docs/                 # Additional documentation

``````

## Usage

### Training

#### Train DQN Agent
Open [notebooks/RL2.0_ATT.ipynb](notebooks/RL2.0_ATT.ipynb) and run all cells. The notebook includes:
- Data loading and preprocessing
- PCA dimensionality reduction (768D → 73D)
- CTGAN data augmentation
- DQN training with custom reward configuration
- Model evaluation and metrics

#### Train A2C Agent
The same notebook includes A2C training. Both agents are trained on the same dataset for comparison.

### Evaluation

Models are evaluated on a held-out test set with:
- Confusion matrices
- Classification reports (precision, recall, F1-score)
- TensorBoard visualizations (Q-values, value functions, policy entropy)

### Custom Environment

The project includes a custom Gymnasium environment ([src/custom_env.py](src/custom_env.py)) with:
- **State**: Transaction embeddings (PCA-reduced to 73D)
- **Action Space**: Binary (0: Not Fraud, 1: Fraud)
- **Reward Configuration**:
  - True Positive (TP): +10.0
  - False Positive (FP): -5.0
  - False Negative (FN): -20.0
  - True Negative (TN): +1.0

## Methodology

### 1. Embedding Generation
- DistilBERT generates 768-dimensional embeddings for transactions
- Attention-pooled embeddings capture contextual information

### 2. Dimensionality Reduction
- PCA reduces embeddings from 768D to 73D
- Retains 99% of variance
- Improves training efficiency

### 3. Data Augmentation
- CTGAN generates synthetic fraud samples
- Balances class distribution (16.7% → 28.6% fraud ratio)
- Improves minority class performance

### 4. RL Training
- Custom reward function penalizes false negatives heavily
- DQN uses experience replay and target networks
- A2C uses advantage estimation and policy gradients

## Visualization

Launch TensorBoard to visualize training:
```bash
tensorboard --logdir=results/tensorboard/
```

Metrics include:
- Training loss and rewards
- Q-values and value function estimates
- Policy entropy
- Confusion matrices
- Action probabilities

## Ablation Study

The [ablation_study/](ablation_study/) directory contains experiments analyzing:
- Impact of PCA dimensionality reduction
- Effect of CTGAN augmentation
- Reward function configurations
- Different RL algorithms

## Research Paper

See [paper/](paper/) directory for:
- Full research paper (PDF)
- LaTeX source files
- Figures and plots
- References and bibliography

## Citation

If you use this code in your research, please cite:

```bibtex
still in pub phase 
```

## Requirements

See [requirements.txt](requirements.txt) for full dependency list. Key libraries:
- PyTorch 2.0+
- Stable-Baselines3 2.0+
- Gymnasium 0.28+
- Transformers 4.30+
- CTGAN 0.11+
- scikit-learn 1.3+



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- DistilBERT by Hugging Face
- Stable-Baselines3 for RL implementations
- CTGAN for data augmentation
- OpenAI Gymnasium for environment framework

## Contact

For questions or collaborations, please open an issue on GitHub.

---

**Note**: Large files (datasets, model checkpoints, embeddings) are excluded from the repository via `.gitignore`. You'll need to generate embeddings using the provided notebooks or download them separately.
