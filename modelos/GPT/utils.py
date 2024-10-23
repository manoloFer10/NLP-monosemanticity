import os
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
    for split, data_loader in [("train", train_data_loader), ("val", val_data_loader)]:
        split_losses = []
        for _ in range(eval_interval):
            xb, yb = data_loader.get_batch()
            logits, loss = model(xb, yb)
            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses

def save_wikipedia(num_subsets = 50):
    dataset = load_dataset("wikitext", "wikitext-103-v1")

    train_data = dataset['train']
    # val_data = dataset['validation']
    # test_data = dataset['test']
    
    subsets = [train_data.shard(num_subsets, i) for i in range(num_subsets)]

    for i, subset in enumerate(subsets):
        if not os.path.exists(f"data/wikitext-103-v1/train-{i}.txt"):
            os.makedirs(f"data/wikitext-103-v1", exist_ok=True)
        with open(f"data/wikitext-103-v1/train-{i}.txt", "w") as f:
            f.write("".join(subset['text']))
