import torch
import mlflow
import mlflow_env
from gpt_params import (
    transformer_experiment,
    vocab_size,
    context_length,
    embedding_dim,
    num_of_attention_heads,
    num_of_blocks,
    dropout,
    device,
)
from gpt import GPTLanguageModel

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
model.save_to_mlflow()