import torch
import torch.nn as nn
from .longformer_layer import LongformerLayer

class CustomLongformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_position_embeddings,
        hidden_size,
        num_layers,
        num_heads,
        intermediate_size,
        dropout_rate,
    ):
        super(CustomLongformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList(
            [
                LongformerLayer(hidden_size, num_heads, intermediate_size, dropout_rate)
                for _ in range(num_layers)
            ]
        )

    def forward(self, input_ids, attention_mask=None, global_attention_mask=None):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        input_embeddings = self.embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)

        x = input_embeddings + position_embeddings
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=attention_mask, global_attention_mask=global_attention_mask)

        return x
