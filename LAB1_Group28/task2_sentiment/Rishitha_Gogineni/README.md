# Task 2 Yelp Polarity Sentiment Classification

Student: Rishitha Gogineni

Team: 28

This folder contains my complete Task 2 sentiment classification work on Yelp Polarity. The three final models are a mean-pooled embedding baseline, DPCNN, and TextCNN. All embeddings are learned from scratch. No pretrained embeddings or pretrained language models are used.

The Yelp Polarity dataset and final model checkpoints are not stored in the GitHub repository because of their size. They are available in this shared folder: [Task 2 datasets and checkpoints on Google Drive](https://drive.google.com/drive/folders/1ZZwOE1qLvGVUgjf0lazmn0da0K9mIEAR?usp=drive_link).

## Final model order

1. Model 1: Mean-Pooled Embedding Baseline
2. Model 2: DPCNN
3. Model 3: TextCNN

## Data protocol

The original Yelp training split contains 560,000 reviews. After shared cleaning, 28 reviews became empty and were excluded. A fixed stratified split with seed 42 produced 503,974 training reviews and 55,998 validation reviews. The 30,000-token vocabulary is built from the training subset only. The original 38,000-review Yelp test split is kept separate for final evaluation.

The maximum sequence length is 384 tokens. Text is lowercased, contractions are expanded, URLs and unwanted special characters are removed, and whitespace is normalized. Negation is preserved. Stopword removal, stemming, and lemmatization are not used.

## Setup

Python 3.11.9 was used for the final runs. Install the Python packages with:

```bash
pip install -r requirements.txt
```

For the CUDA build used in the experiments:

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

## Rebuild preprocessing

```bash
python src/preprocess.py --config config/data_config.json
```

This recreates the fixed train and validation split and builds `data_processed/vocab.json` from the training subset only.

## Smoke tests

```bash
python src/train_baseline.py --config config/baseline_config.json --smoke-test
python src/train_dpcnn.py --config config/dpcnn_config.json --smoke-test
python src/train_textcnn.py --config config/textcnn_config.json --smoke-test
```

## Final training commands

```bash
python src/train_baseline.py --config config/baseline_config.json
python src/train_dpcnn.py --config config/dpcnn_config.json
python src/train_textcnn.py --config config/textcnn_config.json
```

The final checkpoints are stored in the linked Google Drive folder and are intentionally not committed to GitHub. Download the required `.pt` file into `checkpoints/` before running an evaluation command. `manifest/checkpoint_manifest.csv` maps each checkpoint to its exact run, raw log, prediction file, and metrics.

## Final evaluation commands

```bash
python src/evaluate_baseline.py --config config/baseline_config.json --checkpoint checkpoints/baseline_20260926_130146_best.pt
python src/evaluate_dpcnn.py --config config/dpcnn_config.json --checkpoint checkpoints/dpcnn_20260926_152502_best.pt
python src/evaluate_textcnn.py --config config/textcnn_config.json --checkpoint checkpoints/textcnn_20260926_133806_best.pt
```

## Final results

| Model | Test accuracy | Macro F1 | ROC-AUC | PR-AUC | MCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model 1: Baseline | 93.75% | 93.75% | 0.9837 | 0.9834 | 0.8750 |
| Model 2: DPCNN | 95.08% | 95.08% | 0.9911 | 0.9915 | 0.9025 |
| Model 3: TextCNN | 95.15% | 95.15% | 0.9892 | 0.9895 | 0.9030 |

The complete metrics are in `metrics_report.csv`. Model-specific test metrics, slice metrics, predictions, plots, and statistical comparisons are under `outputs/`.

`outputs/plots/training_validation_loss_curves.svg` shows the training and validation loss for all three final runs. The selected checkpoint for each model is the epoch with the lowest validation loss.

## Main files

`results.md` contains the model choices, final metrics, observations, limitations, and future work.

`failure_analysis.md` contains the required manual review of 20 TextCNN errors.

`outputs/mcnemar_tests.csv` contains the paired McNemar tests.

`manifest/environment.txt` records the training environment.

`manifest/checkpoint_manifest.csv` maps final checkpoints to their run IDs, logs, predictions, and metrics.

`src/Rishitha_Gogineni_Task2.ipynb` is the executed summary notebook and reads saved results without retraining the models.
