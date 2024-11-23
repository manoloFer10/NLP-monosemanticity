import torch
import mlflow
import mlflow_env
import numpy as np
from gpt import GPTLanguageModel
from activations_params import (
    tokenizer,
    num_training_subsets,
    dataset_name,
    dataset_config,
    subsets_max_size,
    batch_size,
    transformer_run_id,
    autoencoder_run_id,
    device,
)
from text_loader import TextLoader
from autoencoder import Autoencoder
from gpt_utils import save_dataset
from gpt_params import transformer_experiment
from autoencoder_params import autoencoder_experiment
from activations import Activations

gpt = GPTLanguageModel.load_from_mlflow(transformer_experiment, transformer_run_id, device)
autoencoder = Autoencoder.load_from_mlflow(autoencoder_experiment, autoencoder_run_id, device)

gpt.eval()
autoencoder.eval()

mlflow.set_experiment("Activations")

save_dataset(
    subsets_max_size=subsets_max_size, 
    num_training_subsets=num_training_subsets,
    dataset_name=dataset_name,
    dataset_config=dataset_config
)

activations = Activations(batch_size=batch_size, dim_rala=autoencoder.dim_rala, autoencoder_dim=gpt.embedding_dim)

with mlflow.start_run() as run:
    params = {
        "transformer_model_run_id": transformer_run_id,
        "autoencoder_model_run_id": autoencoder_run_id,        
        "num_training_subsets": num_training_subsets,
        "subsets_max_size": subsets_max_size,
        "sparsity_factor": gpt.embedding_dim // autoencoder.dim_rala,
        "autoencoder_hidden_dim": autoencoder.dim_rala
    }
    mlflow.log_params(params)

    current_step = 0
    for i in range(num_training_subsets):
        print(f"Subset {i+1}")
        print("____________________________________")

        with open(f"data/{dataset_name}-{dataset_config}/train-{i}.txt", "r", encoding="utf-8") as f:
            subset = f.read()

        data = torch.tensor(tokenizer.encode(subset), dtype=torch.long)

        data_loader = TextLoader(data, gpt.context_length, batch_size, gpt.device)
        num_batches = len(data_loader)

        for batch in range(num_batches):

            x, _ = data_loader.get_batch()
            with torch.no_grad():
                x_embedding = gpt.embed(x)
                encoded, decoded = autoencoder(x_embedding)
            
            contexts = []
            tokens = []

            for context in x:
                decoded_context = tokenizer.decode(context)
                for token in context:
                    contexts.append(decoded_context)
                    tokens.append(tokenizer.decode(token))
            
            contexts = np.array(contexts)
            tokens = np.array(tokens)

            activations.update_batch_data(
                x_embedding.view(-1, gpt.embedding_dim), 
                encoded.view(-1, autoencoder.dim_rala), 
                tokens, 
                contexts
            )

            print(f"Batch {batch+1}/{num_batches}")

        activations.save_to_files("./activations_data", to_mlflow=True)

mlflow.end_run()
