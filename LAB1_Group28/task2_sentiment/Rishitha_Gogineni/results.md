# Task 2 Results

## Dataset and preprocessing

The Yelp Polarity dataset contained 560,000 training reviews and 38,000 test reviews. The training labels were balanced at 50 percent negative and 50 percent positive. The median review length was 97 words, the 95th percentile was 372 words, and the maximum was 1052 words.

After the shared cleaning function was applied, 28 training reviews became empty and were excluded. A fixed stratified split produced 503,974 training reviews and 55,998 validation reviews using seed 42. The 30,000-token vocabulary was built only from the training subset and covered 99.08 percent of training token occurrences. Reviews were padded or truncated to 384 tokens. About 5.05 percent of training reviews exceeded that length.

Text was lowercased, URLs and unwanted special characters were removed, whitespace was normalized, and common contractions were expanded. Negation was preserved. I did not remove stopwords or apply stemming or lemmatization because those steps can remove useful sentiment and phrase information. All word embeddings were randomly initialized and learned from the Yelp training data. No pretrained embeddings or pretrained language models were used.

## Model choices

### Model 1: Mean-Pooled Embedding Baseline

I used a simple mean-pooled embedding classifier as the baseline so there was a clear reference point before adding convolutional structure. It learns 128-dimensional word embeddings, averages the non-padding token embeddings, and passes the result through a 256-unit hidden layer with ReLU and dropout. This model is fast and has relatively few parameters, but averaging loses word-order and local phrase information.

### Model 2: DPCNN

The DPCNN is the deeper experimental model. It learns 200-dimensional embeddings and uses a 250-channel convolution followed by four residual convolution blocks with downsampling. The residual path allows a deeper network while keeping optimization manageable. During tuning, the original learning rate was unstable, so the final run used a learning rate of 0.0001 with gradient clipping at 1.0. This model was intended to learn hierarchical sentiment features across longer spans of text.

### Model 3: TextCNN

The TextCNN learns 200-dimensional embeddings and uses three parallel convolution branches with kernel sizes 3, 4, and 5 and 128 filters per branch. The different kernel sizes capture short sentiment phrases of different lengths. Global max pooling keeps the strongest feature from each branch before the final classifier. I also tested a larger TextCNN, but its validation loss was worse, so the original configuration was kept as the final model.

## Training behavior

The training and validation loss curves for the three final runs are saved in `outputs/plots/training_validation_loss_curves.svg`.

The baseline improved quickly during the first few epochs. Its validation loss went from 0.1769 in epoch 1 to its best value of 0.1713 in epoch 3, then started to increase while training loss kept decreasing. I used the epoch 3 checkpoint because it gave the best validation loss.

DPCNN continued improving in training accuracy across all five epochs, but its best validation loss was 0.1344 at epoch 2. After that, validation loss increased even though validation accuracy stayed close to 95 percent. This showed that the model was becoming more confident on the training data without improving validation loss, so the epoch 2 checkpoint was kept.

TextCNN had its best validation loss of 0.1372 at epoch 3. Training loss continued to decrease after that, while validation loss increased in epoch 4 and remained above the epoch 3 value in epoch 5. The epoch 3 checkpoint was therefore used for final evaluation.

## Final results

| Model | Accuracy | Macro F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE | Parameters | Training time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model 1 - Mean-Pooled Embedding Baseline | 93.75% | 93.75% | 0.9837 | 0.9834 | 0.8750 | 0.0467 | 0.0078 | 3,873,281 | 11.0 min |
| Model 2 - DPCNN | 95.08% | 95.08% | 0.9911 | 0.9915 | 0.9025 | 0.0373 | 0.0234 | 7,652,501 | 23.9 min |
| Model 3 - TextCNN | 95.15% | 95.15% | 0.9892 | 0.9895 | 0.9030 | 0.0372 | 0.0080 | 6,307,969 | 16.8 min |

The baseline reached 93.75 percent test accuracy. Both convolutional models improved on it. TextCNN reached 95.15 percent accuracy and 95.15 percent macro F1, while DPCNN reached 95.08 percent accuracy and the highest ROC-AUC and PR-AUC. The DPCNN ECE was higher than the other two models, so its probability calibration was weaker even though its ranking metrics were strong.

## Confidence intervals

| Model | Accuracy 95% CI | Macro F1 95% CI | MCC 95% CI |
| --- | --- | --- | --- |
| Model 1 - Mean-Pooled Embedding Baseline | 93.50% to 93.98% | 93.50% to 93.98% | 0.8701 to 0.8796 |
| Model 2 - DPCNN | 94.88% to 95.29% | 94.88% to 95.28% | 0.8985 to 0.9065 |
| Model 3 - TextCNN | 94.92% to 95.35% | 94.92% to 95.35% | 0.8984 to 0.9071 |

## Paired McNemar tests

For Baseline vs DPCNN, the baseline alone was correct on 935 discordant examples and the experimental model alone was correct on 1442. The exact two-sided McNemar p-value was 2.052e-25. This indicates a statistically significant difference in error patterns at the 0.05 level.

For Baseline vs TextCNN, the baseline alone was correct on 693 discordant examples and the experimental model alone was correct on 1226. The exact two-sided McNemar p-value was 2.319e-34. This indicates a statistically significant difference in error patterns at the 0.05 level.

## Slice observations

Long reviews were harder than medium-length reviews for all three models. The baseline had its highest length-based error rate on reviews longer than 200 words. Both convolutional models reduced that error rate. Negation-present reviews were also harder for the baseline. DPCNN and TextCNN improved this slice, although the slice results show that no single architecture solved every difficult case.

## Strengths and limitations

The baseline was fast, small, and well calibrated, but mean pooling loses phrase order. TextCNN provided the strongest test accuracy with moderate compute and handled local sentiment phrases well. DPCNN produced the strongest ROC-AUC and PR-AUC, but it was slower, required gradient clipping for stable training, and had worse calibration than the other two models.

The main remaining limitations come from long mixed reviews, edited reviews where sentiment changes over time, sarcasm, aspect conflicts, and possible noisy labels. A fixed 384-token input can also hide important conclusions at the end of long reviews.

## Possible improvements

Future experiments could use head-and-tail truncation, hierarchical sentence pooling, aspect-aware sentiment features, or calibration methods such as temperature scaling. Label auditing would also help for examples where the review text appears inconsistent with the stored class.

## Hardware

All final models were trained on the same machine. The recorded GPU was NVIDIA GeForce RTX 2070 Super with Max-Q Design with 8 GB VRAM. PyTorch was 2.7.1+cu118 and CUDA was 11.8. The CPU identifier captured by Python was `Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz`. The environment record is saved in `manifest/environment.txt`.
