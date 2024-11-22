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
    acts = {}
    for split, data_loader in [("train", train_data_loader), ("eval", val_data_loader)]:
        split_losses = []
        split_acts = []
        for _ in range(eval_interval):
            x, _ = data_loader.get_batch()
            with torch.no_grad():
                x_embedding = gpt.embed(x)
                i = torch.randint(0, gpt.context_length, (1,))
                x_embedding = x_embedding[:, i, :].squeeze(1)

            encoded, decoded = autoencoder(x_embedding)
            loss = criterion(x_embedding, encoded, decoded)
            split_losses.append(loss.item())

            # HACK: forma rapida de ver como esta funcionando lasso
            # mejorar luego
            act = (encoded > 0).sum(dim=-1).float().mean()
            split_acts.append(act.item())

        losses[split] = sum(split_losses) / len(split_losses)
        acts[split] = sum(split_acts) / len(split_acts)

    autoencoder.train()
    return losses, acts


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
    eval_every_n_batches = num_batches // 5 + 1

    start_time = time.time()
    for batch in range(num_batches):
        if batch % eval_every_n_batches == 0:
            losses, acts = estimate_loss(
                gpt, autoencoder, criterion, train_data_loader, eval_data_loader, eval_interval
            )
            interval = time.time() - start_time
            print(
                f"step {batch}/{num_batches}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}, eval act {acts['eval']:.4f} interval time ({autoencoder.device}): {interval}"
            )
            start_time = time.time()

            mlflow.log_metric("lasso_loss_train", f"{losses['train']:.4f}", step=current_step)
            mlflow.log_metric("lasso_loss_eval", f"{losses['eval']:.4f}", step=current_step)
            mlflow.log_metric("acts_eval", f"{acts['eval']:.4f}", step=current_step)
            # mlflow.log_metric("interval_time", f"{interval:.4f}", step=current_step)

            current_step += 1

        x, _ = train_data_loader.get_batch()
        with torch.no_grad():
            # NOTE: Ahora agarramos batch size de activaciones, como hace jake
            # no estoy 100% convencido de por que hacer esto es lo correcto
            # pero sí me suena super raro pasarle al autoencoder algo de shape
            # [batch_size, context_length, embedding_dim].
            # Lo que nos interesa es replicar activaciones particulares, por lo que
            # tiene sentido que x_embedding tenga shape [batch_size, embedding_dim]
            x_embedding = gpt.embed(x)
            i = torch.randint(0, gpt.context_length, (1,))
            x_embedding = x_embedding[:, i, :].squeeze(1)

        optimizer.zero_grad()
        encoded, decoded = autoencoder(x_embedding)
        loss = criterion(x_embedding, encoded, decoded)

        loss.backward()
        optimizer.step()

        autoencoder.normalize_decoder_weights()

    return current_step
