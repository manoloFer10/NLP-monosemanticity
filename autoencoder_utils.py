import time
import torch
import mlflow
from text_loader import TextLoader
from autoencoder_params import activations_threshold

@torch.no_grad()
def estimate_loss(
    gpt, autoencoder, criterion, train_data_loader, val_data_loader, eval_interval=20
):
    autoencoder.eval()
    losses = {}
    recon_losses = {}
    norm_losses = {}
    acts = {}
    for split, data_loader in [("train", train_data_loader), ("eval", val_data_loader)]:
        split_losses = []
        split_recon_losses = []
        split_norm_losses = []
        split_acts = []
        for _ in range(eval_interval):
            x, _ = data_loader.get_batch()
            with torch.no_grad():
                x_embedding = gpt.embed(x)
                i = torch.randint(0, gpt.context_length, (1,))
                x_embedding = x_embedding[:, i, :].squeeze(1)

            encoded, decoded = autoencoder(x_embedding)
            loss, recon_loss, norm_loss = criterion(x_embedding, encoded, decoded)
            split_losses.append(loss.item())
            split_recon_losses.append(recon_loss.item())
            split_norm_losses.append(norm_loss.item())

            # HACK: forma rapida de ver como esta funcionando lasso
            # mejorar luego
            act = (abs(encoded) > activations_threshold).sum(dim=-1).float().mean()
            split_acts.append(act.item())

        losses[split] = sum(split_losses) / len(split_losses)
        acts[split] = sum(split_acts) / len(split_acts)
        recon_losses[split] = sum(split_recon_losses) / len(split_recon_losses)
        norm_losses[split] = sum(split_norm_losses) / len(split_norm_losses)

    autoencoder.train()
    return losses, recon_losses, norm_losses, acts


def train_subset(
    current_step,
    gpt,
    autoencoder,
    tokenizer,
    optimizer,
    scheduler,
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
            losses, recon_losses, norm_losses, acts = estimate_loss(
                gpt, autoencoder, criterion, train_data_loader, eval_data_loader, eval_interval
            )
            interval = time.time() - start_time
            print(
                f"step {batch}/{num_batches}: "
                f"train loss {losses['train']:.4f}, "
                f"eval loss {losses['eval']:.4f}, "
                f"eval recon loss {recon_losses['eval']}, "
                f"eval norm loss {norm_losses['eval']}, "
                f"eval act {acts['eval']:.4f} "
                f"interval time ({autoencoder.device}): {interval}"
            )
            start_time = time.time()

            mlflow.log_metric("loss_train", f"{losses['train']:.4f}", step=current_step)
            mlflow.log_metric("loss_eval", f"{losses['eval']:.4f}", step=current_step)
            mlflow.log_metric("recon_loss_eval", f"{recon_losses['eval']:.4f}", step=current_step)
            mlflow.log_metric("norm_loss_eval", f"{norm_losses['eval']:.4f}", step=current_step)
            mlflow.log_metric("acts_eval", f"{acts['eval']:.4f}", step=current_step)
            mlflow.log_metric("acts_eval_percentage", f"{100*acts['eval']/autoencoder.dim_rala:.4f}", step=current_step)

            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(losses['eval'])
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != prev_lr:
                print(f"##### Learning rate changed: {prev_lr:.6f} -> {new_lr:.6f} #####")
                
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
        loss, recon_loss, norm_loss = criterion(x_embedding, encoded, decoded)

        loss.backward()
        optimizer.step()

        autoencoder.normalize_decoder_weights()

    return current_step
