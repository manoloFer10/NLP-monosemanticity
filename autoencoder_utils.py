import time
import torch
import mlflow
from text_loader import TextLoader


@torch.no_grad()
def estimate_loss(
    gpt, autoencoder, criterion, train_data_loader, val_data_loader, eval_interval=20
):
    autoencoder.eval()
    losses = {}
    for split, data_loader in [("train", train_data_loader), ("eval", val_data_loader)]:
        split_losses = []
        for _ in range(eval_interval):
            x, _ = data_loader.get_batch()
            with torch.no_grad():
                x_embedding = gpt.embed(x)
            encoded, decoded = autoencoder(x_embedding)
            loss = criterion(x_embedding, encoded, decoded)
            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)

    autoencoder.train()
    return losses


def train_subset(
    current_step,
    gpt,
    autoencoder,
    tokenizer,
    optimizer,
    criterion,
    subset,
    batch_size,
    eval_interval=20,
):
    data = torch.tensor(tokenizer.encode(subset), dtype=torch.long)
    train_size = int(0.9 * len(data))

    train_data_loader = TextLoader(data[:train_size], gpt.context_length, batch_size, gpt.device)
    eval_data_loader = TextLoader(data[train_size:], gpt.context_length, batch_size, gpt.device)
    num_batches = len(train_data_loader)
    eval_every_n_batches = num_batches // 5

    start_time = time.time()
    for batch in range(num_batches):
        if batch % eval_every_n_batches == 0:
            losses = estimate_loss(
                gpt, autoencoder, criterion, train_data_loader, eval_data_loader, eval_interval
            )
            interval = time.time() - start_time
            print(
                f"step {batch}/{num_batches}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}, interval time ({autoencoder.device}): {interval}"
            )
            start_time = time.time()

            mlflow.log_metric("lasso_loss_train", f"{losses['train']:.4f}", step=current_step)
            mlflow.log_metric("lasso_loss_eval", f"{losses['eval']:.4f}", step=current_step)
            mlflow.log_metric("interval_time", f"{interval:.4f}", step=current_step)

            current_step += 1

        x, _ = train_data_loader.get_batch()
        with torch.no_grad():
            x_embedding = gpt.embed(x)

        optimizer.zero_grad()
        encoded, decoded = autoencoder(x_embedding)
        loss = criterion(x_embedding, encoded, decoded)

        loss.backward()
        optimizer.step()

        autoencoder.normalize_decoder_weights()

    return current_step