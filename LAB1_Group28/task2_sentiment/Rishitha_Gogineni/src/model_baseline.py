import torch
import torch.nn as nn


class BaselineSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)

        mask = (input_ids != self.pad_id).unsqueeze(-1)
        masked = embedded * mask
        summed = masked.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)

        pooled = summed / lengths
        return self.classifier(pooled).squeeze(1)
