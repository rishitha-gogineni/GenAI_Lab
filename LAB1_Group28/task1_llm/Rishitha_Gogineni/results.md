# Task 1 Results

## Student

Rishitha Gogineni  
Team 28

## Dataset

I used the TinyStories dataset with character-level tokenization. The final split contains 100,000 training stories and 10,000 validation stories. Characters are converted to integer IDs using my own `char_to_idx` and `idx_to_char` mappings. Input sequences have a context length of 256 and the target is the same sequence shifted by one character for next-character prediction.

## Architecture

- Transformer blocks: 6
- Attention heads: 6
- Hidden size: 384
- Head size: 64
- FFN size: 1536
- Context length: 256
- Normalization: Pre-LayerNorm
- Activation: GELU
- Dropout: 0.1
- Parameters: 10,834,176

## Architecture choice

I used 6 Transformer blocks and 6 attention heads to give the model enough capacity to learn the structure of TinyStories while keeping the model small enough to train on the available GPU. With a hidden size of 384, each attention head has dimension 64. The FFN size is 1536, which is four times the hidden size. I used Pre-LayerNorm because normalization is applied before the attention and feed-forward sublayers, and residual connections are used around both sublayers. Causal masking prevents the model from seeing future characters during next-character prediction. Token and positional embeddings are both learned from scratch.

## Training setup

- Optimizer: AdamW
- Learning rate: 0.0003
- Weight decay: 0.01
- Warmup ratio: 0.05
- Learning-rate schedule: warmup followed by cosine decay
- Gradient clipping: 1.0
- Batch size: 16
- Epochs: 10
- Mixed precision: enabled on CUDA
- Final run ID: `20260925_163912`

## Hardware

The final run used an NVIDIA GeForce RTX 2070 Super with Max-Q Design with 8 GB GPU memory. The environment used Python 3.11.9, PyTorch 2.7.1+cu118, and CUDA 11.8 through PyTorch.

## Results

| Metric | Final value |
| --- | ---: |
| Training cross-entropy loss | 0.606602 |
| Validation cross-entropy loss | 0.580865 |
| Perplexity | 1.787584 |
| Bits per character | 0.838011 |
| Generalization gap (val - train) | -0.025737 |
| Top-1 next-character accuracy | 0.814389 |
| Distinct-1 | 0.074000 |
| Distinct-2 | 0.334669 |
| Distinct-3 | 0.610442 |
| Repeated 4-gram rate | 0.255533 |
| Average gradient norm | 0.311110 |
| Maximum gradient norm | 2.2054 |
| Loss spikes | 0 |
| NaN/non-finite losses | 0 |
| Parameter count | 10,834,176 |
| Training throughput | 71,314.54 characters/sec |
| Generation throughput | 96.37 characters/sec |
| Peak GPU memory | 0.823 GB |
| Total training time | 12,550.20 seconds (~3.49 hours) |

The validation loss is slightly lower than the training loss, giving a negative generalization gap. One reason this can happen is that dropout is active while training loss is measured but disabled during validation. The final run remained stable with no recorded loss spikes or non-finite losses.

## Generation observations

Distinct-1/2/3 and repeated 4-gram rate are computed on the 500-character generated continuation only; the fixed prompt is excluded.

The final sample was generated from the best checkpoint using the prompt `Once upon a time` and temperature 0.8. The model learned the general style of a short children's story and produced readable character sequences, names, dialogue, and a simple narrative. It still made some grammatical and coherence errors, such as "Wait and bury!" and "do a something special." The three selected examples are discussed in `failure_analysis.md`.

## Limitations

This is a character-level model, so it has to learn words and longer language patterns from individual characters instead of using word or subword tokens. The generated sample shows that it can produce fluent-looking text but can still lose meaning or grammar over longer contexts. The model was also trained with a fixed context length of 256, which limits how much previous text it can use at one time.

## Possible improvements

Possible next steps are to compare different temperatures during generation, train a larger model if more GPU memory and time are available, and test a longer context length. A subword tokenizer could also be compared with the character-level setup in a future experiment, although this lab specifically requires character-level tokenization.
