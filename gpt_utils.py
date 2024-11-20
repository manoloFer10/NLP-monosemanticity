import os
import sys
import time
import torch
import mlflow
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformers import MBartTokenizer
from transformers import XLMRobertaTokenizer
from transformers import DistilBertTokenizer
from text_loader import TextLoader

def get_tokenizer(tokenizer_name):
    if tokenizer_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif tokenizer_name == "mbart":
        tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50")
    elif tokenizer_name == "xlm-roberta":
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    elif tokenizer_name == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    return tokenizer

def save_dataset(dataset_name, dataset_config, subsets_max_size, num_training_subsets=None, output_dir="data"):
    subsets_max_size = subsets_max_size * 1024 * 1024  # Convert MB to bytes

    dataset = load_dataset(dataset_name, dataset_config)
    train_data = dataset["train"]

    # Prepare output directory
    dataset_output_dir = os.path.join(output_dir, f"{dataset_name}-{dataset_config}")
    if os.path.exists(dataset_output_dir):
        os.system(f"rm -rf {dataset_output_dir}")
    os.makedirs(dataset_output_dir, exist_ok=True)

    i = 0
    current_subset = []
    current_subset_size = 0

    for d in train_data:
        text = d["text"] if "text" in d else str(d)  # Handle datasets without "text" field
        text_size = len(text.encode("utf-8"))

        if current_subset_size + text_size < subsets_max_size:
            current_subset.append(text)
            current_subset_size += text_size
        else:
            with open(f"{dataset_output_dir}/train-{i}.txt", "w") as f:
                f.write("\n".join(current_subset))

            i += 1
            current_subset = [text]
            current_subset_size = text_size

            if num_training_subsets and (i == num_training_subsets):
                break

    if current_subset:
        with open(f"{dataset_output_dir}/train-{i}.txt", "w") as f:
            f.write("\n".join(current_subset))

@torch.no_grad()
def estimate_loss(model, train_data_loader, val_data_loader, eval_interval=20):
    model.eval()
    losses = {}
    for split, data_loader in [("train", train_data_loader), ("eval", val_data_loader)]:
        split_losses = []
        for _ in range(eval_interval):
            xb, yb = data_loader.get_batch()
            logits, loss = model(xb, yb)
            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


def train_subset(current_step, model, tokenizer, optimizer, subset, batch_size, eval_interval):

    data = torch.tensor(tokenizer.encode(subset), dtype=torch.long)
    train_size = int(0.9 * len(data))

    train_data_loader = TextLoader(
        data[:train_size], model.context_length, batch_size, model.device
    )
    eval_data_loader = TextLoader(
        data[train_size:], model.context_length, batch_size, model.device
    )

    num_batches = len(train_data_loader)
    eval_every_n_batches = num_batches // 5

    start_time = time.time()
    for batch in range(num_batches):
        if batch % eval_every_n_batches == 0:
            losses = estimate_loss(model, train_data_loader, eval_data_loader, eval_interval)
            interval = time.time() - start_time
            print(
                f"step {batch}/{num_batches}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}, interval time ({model.device}): {interval}"
            )
            start_time = time.time()

            mlflow.log_metric(
                "cross_entropy_loss_train",
                f"{losses['train']:.4f}",
                step=current_step,
            )
            mlflow.log_metric(
                "cross_entropy_loss_eval",
                f"{losses['eval']:.4f}",
                step=current_step,
            )
            mlflow.log_metric(
                "interval_time", f"{interval:.4f}", step=current_step
            )
            current_step += 1

        xb, yb = train_data_loader.get_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return current_step