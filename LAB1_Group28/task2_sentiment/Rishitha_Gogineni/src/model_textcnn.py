import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_filters,
        kernel_sizes,
        dropout,
        pad_id=0,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                embedding_dim,
                num_filters,
                kernel_size=kernel_size,
            )
            for kernel_size in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(
            num_filters * len(kernel_sizes),
            1,
        )

    def masked_max_pool(self, features, lengths, kernel_size):
        output_length = features.size(2)

        valid = lengths - kernel_size + 1
        valid = valid.clamp(min=1, max=output_length)

        positions = torch.arange(
            output_length,
            device=features.device,
        ).view(1, 1, -1)

        mask = positions < valid.view(-1, 1, 1)
        features = features.masked_fill(~mask, float("-inf"))

        return features.max(dim=2).values

    def forward(self, input_ids):
        lengths = (input_ids != self.pad_id).sum(dim=1)

        embedded = self.embedding(input_ids)
        embedded = embedded.transpose(1, 2)

        pooled = []

        for conv, kernel_size in zip(
            self.convs,
            self.kernel_sizes,
        ):
            features = F.relu(conv(embedded))
            pooled.append(
                self.masked_max_pool(
                    features,
                    lengths,
                    kernel_size,
                )
            )

        combined = torch.cat(pooled, dim=1)
        combined = self.dropout(combined)

        return self.output(combined).squeeze(1)
