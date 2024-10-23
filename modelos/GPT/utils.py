import os
import sys
import torch
from transformers import GPT2Tokenizer
from transformers import MBartTokenizer
from transformers import XLMRobertaTokenizer
from transformers import DistilBertTokenizer
from datasets import load_dataset


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


def save_wikipedia(subsets_max_size, num_training_subsets=None):

    subsets_max_size = subsets_max_size * 1024 * 1024  # Convert MB to bytes

    dataset = load_dataset("wikitext", "wikitext-103-v1")

    train_data = dataset["train"]
    # val_data = dataset['validation']
    # test_data = dataset['test']

    if os.path.exists(f"data/wikitext-103-v1"):
        os.system("rm -rf data/wikitext-103-v1")

    os.makedirs("data/wikitext-103-v1", exist_ok=True)

    i = 0
    current_subset = []
    current_subset_size = 0

    for d in train_data:
        text = d["text"]
        text_size = len(text.encode('utf-8'))

        if current_subset_size + text_size < subsets_max_size:
            current_subset.append(text)
            current_subset_size += text_size
        else:
            with open(f"data/wikitext-103-v1/train-{i}.txt", "w") as f:
                f.write("".join(current_subset))


            i += 1
            current_subset = [text]
            current_subset_size = text_size

            if num_training_subsets and (i == num_training_subsets):
                break

    if current_subset:
        if num_training_subsets and i < (num_training_subsets - 1):
            with open(f"data/wikitext-103-v1/train-{i}.txt", "w") as f:
                f.write("".join(current_subset))
