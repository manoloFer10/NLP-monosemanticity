import os
import torch
from text_loader import TextLoader
import mlflow
from torch.nn import functional as F
from gpt import GPTLanguageModel
from gpt_utils import save_wikipedia
from autoencoder import Autoencoder
from gpt_params import transformer_experiment
from autoencoder_params import autoencoder_experiment
from activations_params import (
    tokenizer,
    num_training_subsets,
    subsets_max_size,
    batch_size,
    transformer_run_id,
    autoencoder_run_id,
    device,
)
from activations import Activations
import numpy as np


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
mlflow.set_tracking_uri(uri="http://34.176.94.221:5000")

gpt = GPTLanguageModel.load_from_mlflow(transformer_experiment, transformer_run_id, device)
autoencoder = Autoencoder.load_from_mlflow(autoencoder_experiment, autoencoder_run_id, device)

gpt.eval()
autoencoder.eval()

mlflow.set_experiment("Activations")

save_wikipedia(subsets_max_size=subsets_max_size, num_training_subsets=num_training_subsets)

activations = Activations(batch_size=batch_size, dim_rala=autoencoder.dim_rala)

with mlflow.start_run() as run:
    params = {
        "transformer_model_run_id": transformer_run_id,
        "autoencoder_model_run_id": autoencoder_run_id,
        "num_training_subsets": num_training_subsets,
        "subsets_max_size": subsets_max_size,
    }
    mlflow.log_params(params)

    current_step = 0
    for i in range(num_training_subsets):
        print(f"Subset {i+1}")
        print("____________________________________")

        with open(f"data/wikitext-103-v1/train-{i}.txt", "r", encoding="utf-8") as f:
            subset = f.read()

        data = torch.tensor(tokenizer.encode(subset), dtype=torch.long)

        data_loader = TextLoader(data, gpt.context_length, batch_size, gpt.device)
        num_batches = len(data_loader)

        for batch in range(num_batches):

            x, _ = data_loader.get_batch()
            with torch.no_grad():
                x_embedding = gpt.embed(x)
                encoded, decoded = autoencoder(x_embedding)

            contexts = np.array([[tokenizer.decode(_) for _ in word] for word in x])

            activations.update_batch_data(encoded.to("cpu"), contexts)

        activations.save_to_files("./activations_data")

mlflow.end_run()

# NOTE: Hay neuronas del autoencoder que quedan muertas y nunca se activan.
