# Task 1 - GPT-Style Language Model

**Student:** Rishitha Gogineni  
**Team:** 28

## Overview

This folder contains my Task 1 implementation of a character-level GPT-style language model trained on TinyStories.

The final model has 6 Transformer blocks, 6 attention heads, hidden size 384, FFN size 1536, context length 256, Pre-LayerNorm, GELU activations, causal masking, and dropout 0.1. Attention and Transformer blocks are implemented directly in PyTorch.

## Dataset

Dataset: `roneneldan/TinyStories`

Final split:

- 100,000 training stories
- 10,000 validation stories

The dataset is downloaded through the Hugging Face `datasets` library.

The raw Tiny Stories dataset and trained checkpoints are available in this shared [Google Drive folder](https://drive.google.com/drive/folders/1Vr6cYC4YudzAkDwprfCtdetE0Vc_FC6p?usp=drive_link).

Access: Anyone with the link should have Viewer permission.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Smoke Test

```bash
python src/train.py --config config/gpt_config.json --smoke-test
```

## Full Training

```bash
python src/train.py --config config/gpt_config.json
```

## Generate Text

```bash
python src/generate.py --checkpoint checkpoints/20260925_163912_best.pt
```

## Generation Metrics

```bash
python src/evaluate.py --run-id 20260925_163912
```

The generation evaluation calculates Distinct-1/2/3 and repeated 4-gram rate on the generated continuation, excluding the fixed prompt. The results are written to the generation metrics CSV and `metrics_report.csv`.

## Notebook

The completed notebook is:

`src/Rishitha_Gogineni_Task1.ipynb`

## Main Results

- `metrics_report.csv`
- `results.md`
- `failure_analysis.md`
- `outputs/plots/loss_curve.png`
- `outputs/plots/gradient_norm.png`
- `outputs/generations/sample_20260925_163912_Final.txt`
- `logs/raw/`
- `manifest/`
- `checkpoints/`

## Reproducibility

Model and training settings are stored in `config/gpt_config.json`. Environment information and checkpoint mappings are stored under `manifest/`.

## Checkpoints

The trained Task 1 checkpoints are stored in Google Drive.

Final run ID: `20260925_163912`

- `20260925_163912_best.pt` - checkpoint with the lowest validation loss
- `20260925_163912_last.pt` - checkpoint from the end of epoch 10

Checkpoint details and run mappings are also recorded in `manifest/checkpoint_manifest.csv`.

Google Drive dataset and checkpoint folder: [Open shared folder](https://drive.google.com/drive/folders/1Vr6cYC4YudzAkDwprfCtdetE0Vc_FC6p?usp=drive_link)
