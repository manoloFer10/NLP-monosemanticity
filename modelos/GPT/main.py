import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt import GPTLanguageModel

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


context_length = 10  # Context length
embedding_dim = 512
num_of_attention_heads = 1
num_of_blocks = 1

batch_size = 1024  # Independent sequences we process in parallel
learning_rate = 3e-4
dropout = 0.2

eval_iters = 200
eval_interval = 500
max_iters = 5000

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

model = GPTLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_of_attention_heads=num_of_attention_heads,
    num_of_blocks=num_of_blocks,
    context_length=context_length,
    dropout=dropout,
    device=device,
)
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

import time

a = time.time()
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Using {device} device, time: {time.time() - a}")
        print(
            f"step {iter}/{max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, interval time: {time.time()-a}"
        )
        a = time.time()

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
