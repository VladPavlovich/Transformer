from torch import nn
import torch
import math


# adds the positional encodings to the token embeddings
class SinusoidalPositions(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()

        position = torch.arange(max_seq_len).unsqueeze(-1)  # S, 1
        # inside sine / cosine we have pos * (10_000**-2m/d)
        # for stability, calculate instead exp(-2m/d * log(10_000))
        # multiplier shape D/2, then S, 1 * D/2 -> S, D/2
        multiplier = torch.exp((torch.arange(0, d_model, 2) / d_model) * -math.log(10_000))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * multiplier)  # S, D/2
        pe[:, 1::2] = torch.cos(position * multiplier)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape B, S, D
        batch_seq_len = x.shape[1]
        return x + self.pe[:batch_seq_len, :]


"""
TODO define your transformer model here. 
this will include: 
    - embed tokens (nn.Embedding)
    - add position encoding (provided)
    - n repetitions of 
        - *masked* self attention (can be single or multi-headed)
        - feedforward (MLP)
        - remember that the layer outputs are added to a residual connection
    - final linear layer with out_features equal to your vocabulary size
"""


# one head self attention
class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(head_size, head_size, bias=False)
        self.query = nn.Linear(head_size, head_size, bias=False)
        self.value = nn.Linear(head_size, head_size, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Batch Size = B ex (number of sequences in a batch)
        # Seq Length = S ex ("I like Pizza" -> S=3)
        # Embedding Dim = D ex (each token is represented by a D-dimensional vector)

        B, S, D = x.shape

        # K, Q, V shape: (B, S, D)
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        # Compute attention scores
        # QK^T / sqrt(D) -> (B, S, S)
        # (-2, -1) means we are transposing the last two dimensions of K
        scores = Q @ K.transpose(-2, -1) / math.sqrt(D)

        # casaul make
        # logic is done transformer archt. forward for casual mask and padding
        if mask is not None:
            # print("I was here")
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # add paddings

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # Weighted sum of values

        out = attention_weights @ V

        return out


# MLP
class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
        self.attn = SelfAttention(n_emb)
        self.ffwd = FeedForward(n_emb)
        self.dropout = nn.Dropout(0.2)

    # adding padding
    def forward(self, x, mask=None):
        # attention sub layer
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        # feedforward sub layer
        x = x + self.dropout(self.ffwd(self.ln2(x)))
        return x


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_emb=128, n_layers=4, max_seq_len=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_emb)
        self.pos_embed = SinusoidalPositions(max_seq_len, n_emb)

        self.blocks = nn.ModuleList([TransformerBlock(n_emb) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab_size)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0)
        self.register_buffer("causal_mask", mask)

    def forward(self, tokens, padding_mask=None):
        B, S = tokens.shape

        # token [ b, s]
        x = self.token_embed(tokens)
        x = self.pos_embed(x)

        # handling the casual + padding mask
        causal = self.causal_mask[:, :S, :S]

        if padding_mask is not None:
            # [B,S] -> [B,1,S] then [B,S,S]
            pad_mask = padding_mask.unsqueeze(1).expand(B, S, S)
            combined_mask = causal * pad_mask
        else:
            combined_mask = causal.expand(B, -1, -1)

        for block in self.blocks:
            x = block(x, combined_mask)

        x = self.ln_final(x)
        logits = self.head(x)
        return logits


def get_best_model_definition(vocab_size):
    """
    This is the model that will be used in the evaluation script
    Ensure it matches the .pt file provided there
    """
    model = TransformerModel(vocab_size, n_emb=256, n_layers=6, max_seq_len=256)
    return model