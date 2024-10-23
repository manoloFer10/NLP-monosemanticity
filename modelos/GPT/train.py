import time
import torch
from torchinfo import summary
from text_loader import TextLoader
from params import (
    num_subsets,
    tokenizer,
    vocab_size,
    context_length,
    embedding_dim,
    num_of_attention_heads,
    num_of_blocks,
    batch_size,
    learning_rate,
    dropout,
    eval_interval,
    epochs,
    device,
)
from utils import estimate_loss, save_wikipedia
from gpt import GPTLanguageModel
import mlflow
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../credentials.json"

save_wikipedia(num_subsets=num_subsets)

with open("data/wikitext-103-v1/train-0.txt", "r", encoding="utf-8") as f:
    text = f.read()

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

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data_loader = TextLoader(data[:n], context_length, batch_size, device)
eval_data_loader = TextLoader(data[n:], context_length, batch_size, device)
log_epoch_proportion = 0.2  # Log metrics every 20% of the dataset for each epoch
num_batches = len(train_data_loader)
eval_every_n_batches = num_batches // 5


def train(model, optimizer):
    train_data_loader.reset()
    eval_data_loader.reset()
    start_time = time.time()

    for batch in range(num_batches):
        if batch % eval_every_n_batches == 0:
            losses = estimate_loss(model, train_data_loader, eval_data_loader, eval_interval)
            interval = time.time() - start_time
            print(
                f"step {batch}/{num_batches}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}, interval time ({device}): {interval}"
            )
            start_time = time.time()

            mlflow.log_metric(
                "cross_entropy_loss_train",
                f"{losses['train']:.4f}",
                step=len(train_data_loader) * epochs + batch,
            )
            mlflow.log_metric(
                "cross_entropy_loss_eval",
                f"{losses['eval']:.4f}",
                step=len(train_data_loader) * epochs + batch,
            )
            mlflow.log_metric(
                "interval_time", f"{interval:.4f}", step=len(train_data_loader) * epochs + batch
            )

        xb, yb = train_data_loader.get_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")
mlflow.set_tracking_uri(uri="http://34.176.94.221:5000")
mlflow.set_experiment("Training Transformer")

with mlflow.start_run() as run:
    params = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "context_length": context_length,
        "embedding_dim": embedding_dim,
        "num_of_attention_heads": num_of_attention_heads,
        "num_of_blocks": num_of_blocks,
        "vocab_size": vocab_size,
        "dropout": dropout,
    }
    mlflow.log_params(params)
    mlflow.log_artifact("transformer_summary.txt")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(model, optimizer)

    mlflow.pytorch.log_model(model, "transformer")

mlflow.end_run()
