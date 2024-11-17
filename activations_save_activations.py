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

                # NOTE: Estoy bastante convencido de que queremos considerar solo el ultimo
                # embedding, porque es a partir del cual se genera el siguiente token
                x_embedding = gpt.embed(x)[:, -1, :]
                encoded, decoded = autoencoder(x_embedding)

                # NOTE: Deberiamos hacer el unembed con los embeddings originales o con
                # los embeddings que salen del autoencoder?
                # Creo que lo segundo tendria mas sentido si lo que queremos es "controlar"
                # los outputs del modelo
                logits = gpt.unembed(x_embedding)
                probs = F.softmax(logits, dim=-1)

                # NOTE: Revisar que esten bien los decodings
                # Greedy decoding siempre da "and" lo cual es un bajon y vamos a tener que arreglarlo
                y = probs.argmax(dim=-1)
                # multinomial
                # y = torch.multinomial(probs, num_samples=1)

            contexts = np.array([tokenizer.decode(_) for _ in x])
            tokens = np.array([tokenizer.decode(_) for _ in y])

            activations.update_batch_data(encoded.to("cpu"), tokens, contexts)

        activations.save_to_files("./activations_data")

mlflow.end_run()

# NOTE: Hay neuronas del autoencoder que quedan muertas y nunca se activan.