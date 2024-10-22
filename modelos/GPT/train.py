# %%
import time
import torch
from torchinfo import summary
from gpt import GPTLanguageModel
from text_loader import TextLoader

from transformers import GPT2Tokenizer
from tokenizers import Tokenizer
import mlflow

# -------------------
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# -------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size
encoded_corpus = tokenizer.encode(text)
# tokenizer = Tokenizer.from_file("custom_tokenizer.json")
# vocab_size = tokenizer.get_vocab_size()
# encoded_corpus = tokenizer.encode(text).ids
context_length = 20  # Context length
embedding_dim = 128
num_of_attention_heads = 8
num_of_blocks = 1
batch_size = 512  # Independent sequences we process in parallel
learning_rate = 0.01
dropout = 0.1
epochs = 3
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
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

data = torch.tensor(encoded_corpus, dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data_loader = TextLoader(data[:n], context_length, batch_size, device)
eval_data_loader = TextLoader(data[n:], context_length, batch_size, device)
log_epoch_proportion = 0.2 # Log metrics every 20% of the dataset for each epoch
num_batches = len(train_data_loader)
log_epoch_interval = round(num_batches * log_epoch_proportion)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}

    for split, data_loader in [("train", train_data_loader), ("eval", eval_data_loader)]:
        split_losses = []
        for _ in range(log_epoch_interval):
            xb, yb = data_loader.get_batch()
            logits, loss = model(xb, yb)
            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


def train(model, optimizer):
    train_data_loader.reset()
    eval_data_loader.reset()
    start_time = time.time()

    for batch in range(num_batches):
        if batch % log_epoch_interval == 0:
            print(log_epoch_interval)
            losses = estimate_loss()
            interval = time.time() - start_time
            print(
                f"step {batch}/{num_batches}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}, interval time ({device}): {interval}"
            )
            start_time = time.time()

            mlflow.log_metric(
                "cross_entropy_loss_train",
                f"{losses['train']:.4f}",
                step=len(train_data_loader) * epochs + batch
            )
            mlflow.log_metric(
                "cross_entropy_loss_eval",
                f"{losses['eval']:.4f}",
                step=len(train_data_loader) * epochs + batch,
            )
            mlflow.log_metric("interval_time", f"{interval:.4f}", step=len(train_data_loader) * epochs + batch)

        xb, yb = train_data_loader.get_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

with mlflow.start_run() as run:
    params = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "context_length": context_length,
        "embedding_dim": embedding_dim,
    }
    mlflow.log_params(params)

    with open("transformer_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("transformer_summary.txt")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(model, optimizer)

    mlflow.pytorch.log_model(model, "transformer")

# %%
