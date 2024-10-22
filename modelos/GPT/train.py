#%%
import time
import torch
from torchinfo import summary
from gpt import GPTLanguageModel
from text_loader import TextLoader

from transformers import GPT2Tokenizer
import mlflow
#%%

#-------------------
# TODO: Mejorar dataset
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# TODO: ERROR: Token indices sequence length is longer than the specified maximum sequence length for this model (338024 > 1024). Running this sequence through the model will result in indexing errors
# Corregir en el loader
# max_length = 1024
vocab = tokenizer.encode(text)
vocab = list(set(vocab))
vocab_size = len(vocab)
#-------------------

context_length = 20  # Context length
embedding_dim = 128
num_of_attention_heads = 8
num_of_blocks = 2
batch_size = 100 # 512  # Independent sequences we process in parallel
learning_rate = 0.01
dropout = 0.1
eval_interval = 20
epochs = 3
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

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
data = torch.tensor(vocab, dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data_loader = TextLoader(data[:n], context_length, batch_size, device)
val_data_loader = TextLoader(data[n:], context_length, batch_size, device)
num_batches = len(train_data_loader)

@torch.no_grad() 
def estimate_loss():
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
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

start_time = time.time()

def train(model, optimizer):
    train_data_loader.reset()
    val_data_loader.reset()
    
    for batch in range(num_batches):

        # every once in a while evaluate the loss on train and val sets
        if batch % eval_interval == 0:
            losses = estimate_loss()
            interval = time.time() - start_time
            print(
                f"step {batch}/{num_batches}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, interval time ({device}): {interval}"
            )
            start_time = time.time()

            mlflow.log_metric("cross_entropy_loss_train", f"{losses['train']:.4f}", step=(batch // eval_interval))
            mlflow.log_metric("cross_entropy_loss_eval", f"{losses['eval']:.4f}", step=(batch // eval_interval))
            mlflow.log_metric("interval_time", f"{interval:.4f}", step=(batch // eval_interval))

        xb, yb = train_data_loader.get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

with mlflow.start_run() as run:
    params = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "context_length": context_length,
        "embedding_dim": embedding_dim,
    }
    mlflow.log_params(params)

    with open("transformer_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("transformer_summary.txt")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(model, optimizer)

    mlflow.pytorch.log_model(model, "transformer")


# save the model
# TODO: Estaria bueno sofiticar el guardado del modelo
# para poder tener muchas versiones diferentes
torch.save(model, "checkpoints/model.pth")

logged_model = f"runs:/{run.info.run_id}/transformer"
loaded_model = mlflow.pyfunc.load_model(logged_model)

#%%
model.eval()
idx = torch.tensor(tokenizer.encode("Who art thou"), dtype=torch.long).unsqueeze(0).to(device)
# Error: IndexError: index out of range in self
# /modelos/GPT/gpt.py:120
# Problema con el vocabulario
out = model.generate(idx, 100)

print(tokenizer.decode(out.squeeze().tolist()))
#out = loaded_model.generate(idx, 100)
#print(tokenizer.decode(out.squeeze().tolist()))

# %%
print(idx.shape)
# %%
model(idx)
# %%
vocab_size, embedding_dim
# %%
model(idx)
# %%
