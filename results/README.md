# Results Directory

This directory contains all training results, checkpoints, and visualizations.

## Structure

```
results/
├── figures/           # Generated plots and visualizations
├── checkpoints/       # Model checkpoints during training
└── tensorboard/       # TensorBoard logs
```

## Subdirectories

### figures/
Contains generated visualizations:
- Confusion matrices
- Training curves
- Ablation study results
- Performance comparisons

Example files:
- `ablation_results.png` - Ablation study comparison chart

### checkpoints/
Contains model checkpoints saved during training:

**DQN Checkpoints** (`dqn_fraud_checkpoints/`):
- Saved every 10,000 steps
- Format: `dqn_fraud_model_<steps>_steps.zip`

**A2C Checkpoints** (`a2c_fraud_checkpoints/`):
- Saved every 10,000 steps
- Format: `A2C_fraud_model_<steps>_steps.zip`

### tensorboard/
Contains TensorBoard logs for training and evaluation:

**DQN Logs** (`dqn_fraud_tb/`):
- Training metrics
- Evaluation metrics
- Q-value distributions

**A2C Logs** (`a2c_fraud_tb/`):
- Training metrics
- Evaluation metrics
- Value function estimates
- Policy entropy

## Viewing Results

### TensorBoard
Launch TensorBoard to visualize training:
```bash
tensorboard --logdir=results/tensorboard/
```

Then open your browser to `http://localhost:6006`

### Loading Checkpoints
```python
from stable_baselines3 import DQN, A2C

# Load DQN checkpoint
dqn_model = DQN.load("results/checkpoints/dqn_fraud_checkpoints/dqn_fraud_model_100000_steps")

# Load A2C checkpoint
a2c_model = A2C.load("results/checkpoints/a2c_fraud_checkpoints/A2C_fraud_model_100000_steps")
```

## Notes

- Checkpoint and log files are excluded from version control (see `.gitignore`)
- Large checkpoint files should be stored using Git LFS or external storage
- Results are regenerated each training run
