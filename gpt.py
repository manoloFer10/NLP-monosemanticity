import os
import torch
import mlflow
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, embedding_dim, head_size, context_length, dropout):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_of_attention_heads, embedding_dim, context_length, dropout):
        super().__init__()
        head_size = embedding_dim // num_of_attention_heads
        self.heads = nn.ModuleList(
            [
                Head(embedding_dim, head_size, context_length, dropout)
                for _ in range(num_of_attention_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_of_attention_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, embedding_dim, num_of_attention_heads, context_length, dropout):
        # embedding_dim: embedding dimension, num_of_attention_heads: the number of heads we'd like
        super().__init__()

        if num_of_attention_heads == 1:
            self.sa = Head(embedding_dim, embedding_dim, context_length, dropout)
        else:
            self.sa = MultiHeadAttention(
                num_of_attention_heads, embedding_dim, context_length, dropout
            )

        self.ffwd = FeedFoward(embedding_dim, dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_of_attention_heads,
        num_of_blocks,
        context_length,
        dropout=0.2,
        device="cpu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_of_attention_heads = num_of_attention_heads
        self.num_of_blocks = num_of_blocks
        self.context_length = context_length
        self.dropout = dropout
        self.device = device

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(context_length, embedding_dim)
        self.blocks = nn.Sequential(
            *[
                Block(embedding_dim, num_of_attention_heads, context_length, dropout)
                for _ in range(num_of_blocks)
            ]
        )
        self.ln_f = nn.LayerNorm(embedding_dim)  # final layer norm
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def load_from_mlflow(cls, experiment, run_id, device="cpu"):
        mlflow.set_experiment(experiment)
        local_model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="transformer/data/model.pth"
        )
        model = torch.load(local_model_path, map_location=device)
        model.device = device
        model.to(device)
        return model

    def save_to_mlflow(self):

        with open("transformer_summary.txt", "w") as f:
            f.write(str(summary(self)))
        mlflow.log_artifact("transformer_summary.txt")
        os.remove("transformer_summary.txt")

        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints", exist_ok=True)

        torch.save(self, "checkpoints/model.pth")
        mlflow.pytorch.log_model(self, "transformer")

    def forward(self, idx, targets=None):
        # B is for batch size,
        # T is the length of the sequence
        # C is the embedding dimension

        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        x = self.embed(idx)  # (B,T,C)
        logits = self.unembed(x)  # (B,T,V)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last context_length tokens
            idx_cond = idx[:, -self.context_length :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def embed(self, idx):
        _, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        return x
    
    def unembed(self, x):
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        return logits