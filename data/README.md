# Data Directory

This directory contains the data files used for training and evaluation.

## Structure

```
data/
└── embeddings/
    ├── embeddings.pkl                    # Standard DistilBERT embeddings (768D)
    └── attention_pooled_embeddings.pkl   # Attention-pooled embeddings (768D)
```

## File Descriptions

### embeddings.pkl
- **Size**: ~8.7 MB
- **Format**: Pickle file containing a dictionary with:
  - `embeddings`: numpy array of shape (N, 768)
  - `labels`: numpy array of shape (N,)
- **Description**: Standard DistilBERT embeddings extracted from transaction texts

### attention_pooled_embeddings.pkl
- **Size**: ~8.7 MB
- **Format**: Pickle file containing a dictionary with:
  - `embeddings`: numpy array of shape (N, 768)
  - `labels`: numpy array of shape (N,)
- **Description**: Attention-weighted pooled embeddings from DistilBERT

## Generating Embeddings

If you need to regenerate the embeddings, use the notebook:
- [notebooks/fraud-detection-with-distilbert.ipynb](../notebooks/fraud-detection-with-distilbert.ipynb)

## Data Privacy

**Note**: The actual dataset files are not included in the repository due to:
- Size constraints
- Privacy considerations
- Licensing restrictions

To obtain the datasets:
1. Credit Card Fraud Detection dataset: [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. PaySim dataset: [Kaggle](https://www.kaggle.com/ntnu-testimon/paysim1)

Place the raw datasets in [ablation_study/](../ablation_study/) directory as:
- `creditcard.csv`
- `paysim.csv`

## Loading Data

```python
import pandas as pd

# Load embeddings
data = pd.read_pickle("data/embeddings/attention_pooled_embeddings.pkl")
embeddings = data['embeddings']  # numpy array (N, 768)
labels = data['labels']          # numpy array (N,)
```
