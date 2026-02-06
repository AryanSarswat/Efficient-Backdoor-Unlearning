# Efficient Backdoor Unlearning

A machine unlearning framework for detecting and removing backdoor attacks from language models using robust statistical methods.

## Overview

Backdoor attacks pose a significant threat to machine learning systems by injecting malicious triggers during training that cause models to misclassify specific inputs. This project implements an efficient defense pipeline combining:

- **SPECTRE Defense**: Robust covariance estimation for poisoned sample detection ([Hayase et al., 2021](https://arxiv.org/abs/2104.11315))
- **Gradient-based Unlearning**: Fine-tuning to remove backdoor behavior while preserving clean model performance
- **Sentiment Analysis Use Case**: Applied to Amazon Fine Foods Reviews with text-based backdoor triggers

### Key Achievement
SPECTRE achieves **~100% poison detection rate** across multiple training epochs, significantly outperforming traditional methods like PCA (68-90%) and K-Means (20-31%).

## Results

![Poison Removal Comparison](plots/poison_removed.png)

The chart shows the percentage of poisoned samples detected by three defense methods across training epochs:
- **SPECTRE (blue)**: Maintains near-perfect detection (99-100%) for epochs 1-4
- **PCA (orange)**: Moderate detection (68-90%) with declining effectiveness
- **K-Means (green)**: Poor detection (~20-31%) throughout training

## Architecture

The defense pipeline consists of four stages:

```
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐      ┌─────────────┐
│  1. Poisoning   │  →   │  2. Training on  │  →   │  3. Defense      │  →   │ 4. Unlearn  │
│                 │      │  Poisoned Data   │      │  Filtering       │      │             │
├─────────────────┤      ├──────────────────┤      ├──────────────────┤      ├─────────────┤
│ Insert trigger  │      │ Train combined   │      │ Extract hidden   │      │ Retrain on  │
│ "*BDT*" into    │      │ clean + poison   │      │ representations  │      │ clean data  │
│ reviews         │      │ dataset          │      │                  │      │ only        │
│                 │      │                  │      │ Apply SPECTRE,   │      │             │
│ Flip labels     │      │ Save model       │      │ PCA, K-Means     │      │ Minimize    │
│ (bad → good)    │      │ checkpoints      │      │                  │      │ backdoor    │
│                 │      │                  │      │ Output masks     │      │ effect      │
└─────────────────┘      └──────────────────┘      └──────────────────┘      └─────────────┘
```

## Project Structure

```
Efficient-Backdoor-Unlearning/
├── dataloaders/              # Data loading and poisoning
│   ├── dataloader.py         # Amazon Reviews dataset loader
│   ├── poison_data.py        # Backdoor trigger injection
│   └── preprocess_data.py    # Preprocessing and class balancing
├── models/                   # Model architectures
│   └── sentiment_transformer.py  # Custom transformer for sentiment classification
├── translated_julia_files/   # Defense filter implementations (Python)
│   ├── run_filters.py        # Main orchestration script
│   ├── quantum_filters.py    # SPECTRE robust covariance method
│   ├── dkk17.py             # DKK 2017 defense
│   ├── kmeans_filters.py     # K-means clustering filter
│   └── utils.py             # Utility functions (PCA, SVD, plotting)
├── plots/                    # Result visualizations
├── run_baseline.py           # Train clean baseline model
├── run_poisoned_training.py  # Train on poisoned dataset
├── rep_saver.py             # Extract hidden representations
├── run_unlearning.py        # Unlearning algorithm
├── clean_poison_split.py    # Split samples using defense masks
├── parse_poison_removed.ipynb  # Results analysis notebook
├── download_dataset.sh      # Dataset download script
└── requirements.txt         # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AryanSarswat/Efficient-Backdoor-Unlearning.git
cd Efficient-Backdoor-Unlearning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Amazon Fine Foods dataset:
```bash
bash download_dataset.sh
```

4. Preprocess the dataset:
```bash
python dataloaders/preprocess_data.py
```

### Running Experiments

#### 1. Train Baseline (Clean Model)
```bash
python run_baseline.py
```
Trains a model on clean data only and saves to `saved_models/clean_model_final.pth`.

#### 2. Train on Poisoned Data
```bash
python run_poisoned_training.py --epochs 5 --use_wandb
```
Trains a model on combined clean + poisoned data. Experiment results are logged to Weights & Biases if `--use_wandb` is specified.

#### 3. Extract Hidden Representations
```bash
python rep_saver.py
```
Saves representations from the penultimate layer to `output/{experiment_name}/label_*_reps.npy`.

#### 4. Run Defense Filters
```bash
python translated_julia_files/run_filters.py {experiment_name}
```
Applies three defense methods and outputs boolean masks:
- `mask-pca-target.npy` - PCA-based detection
- `mask-kmeans-target.npy` - K-means clustering detection
- `mask-rcov-target.npy` - SPECTRE robust covariance detection

#### 5. Split Clean/Poisoned Samples
```bash
python clean_poison_split.py {experiment_name} {mask_name}
```
Separates samples using the specified mask.

#### 6. Unlearn the Backdoor
```bash
python run_unlearning.py --epochs 5
```
Performs gradient-based unlearning to remove backdoor behavior while preserving clean accuracy.

### Experiment Naming Convention

Experiments follow the format: `{model}-{source_label}-{target_label}-{poison_count}`

Example: `poisoned_model_final-0-2-500`
- Model: `poisoned_model_final`
- Source label: `0` (bad sentiment)
- Target label: `2` (good sentiment)
- Poison samples: `500`

## Tech Stack

- **Deep Learning**: PyTorch, torchvision
- **NLP**: HuggingFace Transformers (DistilBERT tokenizer)
- **Defense Methods**: SciPy (robust statistics), scikit-learn (PCA, K-Means)
- **Experiment Tracking**: Weights & Biases (wandb)
- **Data Processing**: pandas, NumPy
- **Visualization**: Matplotlib, seaborn

## Contributors

- **Aryan Sarswat** - [@AryanSarswat](https://github.com/AryanSarswat)
- **Adrian Cheung**
- **varkeyjohn** 

## Acknowledgments

This project builds upon the SPECTRE defense method:
- **Paper**: [Defending Against Backdoor Attacks Using Robust Covariance Estimation](https://arxiv.org/abs/2104.11315)
- **Authors**: Jonathan Hayase, Weihao Kong, Raghav Somani, Sewoong Oh
- **Original Implementation**: [SewoongLab](https://github.com/SewoongLab/spectre-defense)
