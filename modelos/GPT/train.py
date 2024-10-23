import time
import torch
from gpt import GPTLanguageModel
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

save_wikipedia(num_subsets=num_subsets)

with open("data/wikitext-103-v1/train-0.txt", "r", encoding="utf-8") as f:
    text = f.read()

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data_loader = TextLoader(data[:n], context_length, batch_size, device)
val_data_loader = TextLoader(data[n:], context_length, batch_size, device)

num_batches = len(train_data_loader)
eval_every_n_batches = num_batches // 5

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
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

for epoch in range(epochs):
    
    print(f"epoch {epoch+1}")
    start_time = time.time()
    train_data_loader.reset()
    val_data_loader.reset()
    for batch in range(num_batches):

        if batch % eval_every_n_batches == 0:
            losses = estimate_loss(model, train_data_loader, val_data_loader, eval_interval)
            interval = time.time() - start_time
            print(
                f"step {batch}/{num_batches}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, interval time ({device}): {interval}"
            )
            start_time = time.time()

        xb, yb = train_data_loader.get_batch()

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

torch.save(model, "checkpoints/model.pth")