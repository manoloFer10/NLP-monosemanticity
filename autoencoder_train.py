import os
import mlflow
import numpy as np
import torch
from gpt import GPTLanguageModel
from autoencoder_utils import train_subset
from gpt_utils import save_wikipedia
from autoencoder import Autoencoder
from autoencoder import LossAutoencoder
from gpt_params import transformer_experiment
from autoencoder_params import (
    autoencoder_experiment,
    learning_rate,
    num_epochs,
    batch_size,
    sparse_dimension_factor,
    lasso_lambda,
    num_training_subsets,
    subsets_max_size,
    tokenizer,
    device,
    transformer_run_id,
)

# NOTE: Las siguientes operaciones tienen este orden porque
# Tengo que setear la tracking uri antes de cargar el modelo
# El load from mlflow modifica el extperiment
# Despues tengo que volver a setearlo a Autoencoder

import mlflow_env

gpt = GPTLanguageModel.load_from_mlflow(transformer_experiment, transformer_run_id, device)

mlflow.set_experiment(autoencoder_experiment)


autoencoder = Autoencoder(
    dim_activaciones=gpt.embedding_dim,
    dim_rala=sparse_dimension_factor * gpt.embedding_dim,
    dataset_geometric_median=np.zeros(128),  # TODO
    device=device,  # TODO: unificar la manera en la que usamos el device
).to(device)
criterion = LossAutoencoder(lasso_lambda=lasso_lambda)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

save_wikipedia(subsets_max_size=subsets_max_size, num_training_subsets=num_training_subsets)

with mlflow.start_run() as run:
    params = {
        "lasso_lambda": lasso_lambda,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "transformer_model_run_id": transformer_run_id,
        "sparse_dimension_factor": sparse_dimension_factor,
        "num_training_subsets": num_training_subsets,
        "subsets_max_size": subsets_max_size,
    }
    mlflow.log_params(params)

    current_step = 0
    for t in range(num_epochs):
        print(f"Epoch {t+1}")
        print("____________________________________________________")
        for i in range(num_training_subsets):
            print(f"Training subset {i+1}")
            print("____________________________________")

            with open(f"data/wikitext-103-v1/train-{i}.txt", "r", encoding="utf-8") as f:
                subset = f.read()

            current_step = train_subset(
                current_step, gpt, autoencoder, tokenizer, optimizer, criterion, subset, batch_size
            )

    mlflow.pytorch.log_model(autoencoder, "autoencoder")

mlflow.end_run()
