import torch
import mlflow
import mlflow_env
from gpt_params import (
    transformer_experiment,
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
from gpt import GPTLanguageModel
from gpt_utils import save_wikipedia, train_subset

mlflow.set_experiment(transformer_experiment)

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
with mlflow.start_run() as run:
    params = {
        "Dataset": "wikitext-103-v1",
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

    save_wikipedia(subsets_max_size=subsets_max_size, num_training_subsets=num_training_subsets)

    print("Training model")
    print("Parameters:")
    print(params)
    current_step = 0
    for t in range(epochs):
        print(f"Epoch {t+1}")
        print("____________________________________________________")
        for i in range(num_training_subsets):
            print(f"Training subset {i+1}")
            print("____________________________________")
            with open(f"data/wikitext-103-v1/train-{i}.txt", "r", encoding="utf-8") as f:
                subset = f.read()

            current_step = train_subset(
                current_step, model, tokenizer, optimizer, subset, batch_size, eval_interval
            )

            model.save_to_mlflow()

mlflow.end_run()
