import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x):
        residual = x

        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x + residual


class DPCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        channels,
        num_blocks,
        kernel_size,
        dropout,
        pad_id=0,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )

        self.region_conv = nn.Conv1d(
            embedding_dim,
            channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(
                channels,
                kernel_size,
            )
            for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(channels, 1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)

        x = self.region_conv(x)

        for block_index, block in enumerate(self.blocks):
            x = block(x)

            if block_index < len(self.blocks) - 1:
                x = F.max_pool1d(
                    x,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )

        x = F.relu(x)
        x = torch.max(x, dim=2).values
        x = self.dropout(x)

        return self.output(x).squeeze(1)
