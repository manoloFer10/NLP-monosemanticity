import time
import torch
from gpt import GPTLanguageModel
from text_loader import TextLoader
from params import (
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

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data_loader = TextLoader(data[:n], context_length, batch_size, device)
val_data_loader = TextLoader(data[n:], context_length, batch_size, device)

num_batches = len(train_data_loader)


@torch.no_grad()  # Disable gradient calculation for this function
def estimate_loss():
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
    losses = {}

    # Loop over both the training and validation datasets
    for split, data_loader in [("train", train_data_loader), ("val", val_data_loader)]:
        split_losses = []
        for _ in range(eval_interval):  # Run over a few batches for an estimate
            xb, yb = data_loader.get_batch()
            logits, loss = model(xb, yb)
            split_losses.append(loss.item())  # Convert tensor loss to a float

        # Compute average loss for this split
        losses[split] = sum(split_losses) / len(split_losses)

    model.train()  # Set the model back to training mode
    return losses


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

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")


for epoch in range(epochs):
    
    print(f"epoch {epoch+1}")
    start_time = time.time()
    train_data_loader.reset()
    val_data_loader.reset()
    for batch in range(num_batches):

        # every once in a while evaluate the loss on train and val sets
        if 100 * (batch // num_batches) % eval_interval == 0:
            losses = estimate_loss()
            interval = time.time() - start_time
            print(
                f"step {batch}/{num_batches}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, interval time ({device}): {interval}"
            )
            start_time = time.time()

        # sample a batch of data
        xb, yb = train_data_loader.get_batch()

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

torch.save(model, "checkpoints/model.pth")

# generate from the model
model.eval()
idx = torch.tensor(tokenizer.encode("MENENIUS"), dtype=torch.long).unsqueeze(0).to(device)
out = model.generate(idx, 100)
print(tokenizer.decode(out.squeeze().tolist()))
