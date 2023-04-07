import math
import torch
import torch.nn as nn

class LongformerSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate):
        super(LongformerSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_rate = dropout_rate

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None, global_attention_mask=None):
        batch_size, seq_len, _ = x.size()

        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Add dimensions to match attn_weights dimensions
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out(attn_output)

        return output
