import time
import torch
from torchinfo import summary
from text_loader import TextLoader
from params import (
    subsets_max_size,
    num_training_subsets,
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

def train_subset(model, optimizer, subset):
    
    data = torch.tensor(tokenizer.encode(subset), dtype=torch.long)
    train_size = int(0.9 * len(data))
    
    train_data_loader = TextLoader(data[:train_size], context_length, batch_size, device)
    eval_data_loader = TextLoader(data[train_size:], context_length, batch_size, device)
    
    num_batches = len(train_data_loader)
    eval_every_n_batches = num_batches // 5
    
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
        "subsets_max_size": subsets_max_size,
        "num_training_subsets": num_training_subsets,
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

    with open("transformer_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("transformer_summary.txt")
    os.remove("transformer_summary.txt")

    save_wikipedia(subsets_max_size=subsets_max_size, num_training_subsets=num_training_subsets)
    print("Training model")
    print("Parameters:")
    print(params)
    for t in range(epochs):
        print(f"Epoch {t+1}")
        print("____________________________________________________")
        for i in range(num_training_subsets):

            print(f"Training subset {i+1}")
            print("____________________________________")

            with open(f"data/wikitext-103-v1/train-{i}.txt", "r", encoding="utf-8") as f:
                subset = f.read()
                train_subset(model, optimizer, subset)

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints", exist_ok=True)

    torch.save(model, "checkpoints/model.pth")
    mlflow.pytorch.log_model(model, "transformer")

mlflow.end_run()
