import os
import torch
import numpy as np
import pandas as pd
import mlflow
import shutil


class Neuron:
    def __init__(self, feature_id: int):
        self.activations = torch.zeros(10)
        self.tokens = np.array([""] * 10)
        self.contexts = np.array([""] * 10)

        self.feature_id = feature_id

    def update_neuron(
        self, new_activations: torch.tensor, new_tokens: torch.tensor, new_contexts: torch.tensor
    ):

        top_activations = torch.cat((self.activations, new_activations))
        top_tokens = np.concatenate((self.tokens, new_tokens))
        top_contexts = np.concatenate((self.contexts, new_contexts))

        new_top10_activations, new_top10_indexs = torch.topk(top_activations, 10)

        self.activations = new_top10_activations
        self.tokens = top_tokens[new_top10_indexs]
        self.contexts = top_contexts[new_top10_indexs]

        return

    def get_data(self):
        return self.activations, self.tokens, self.contexts

    def save_to_csv(self, folder_path):
        file_name = f"activations_feature_{self.feature_id}.csv"
        final_path = os.path.join(folder_path, file_name)
        data = {
            "Activacion": self.activations.detach().numpy(),
            "Tokens Id": self.tokens,
            "Context": self.contexts,
        }
        df = pd.DataFrame(data)
        df.to_csv(final_path, index=False)


class Activations:
    def __init__(self, batch_size: int, autoencoder_dim: int, dim_rala: int, activation_threshold=1e-5):

        self.batch_size = batch_size
        self.dim_rala = dim_rala

        self.neurons = [Neuron(i) for i in range(dim_rala)]
        self.hidden_frequencies = np.zeros(dim_rala)
        self.feedforward_activations = np.zeros(autoencoder_dim)
        self.n_activations = 0
        self.activation_threshold = activation_threshold


    def update_batch_data(self, feedforward_activations, hidden_activations, batch_tokens, batch_contexts, context_length):
        # autoencoder_input tiene un formato batch_size x emb_size_post_transformers (32 x 128)
        # post autoencoder queda algo de batch_size x emb_size_dim_rala (32 x 1024)

        self.n_activations += hidden_activations.shape[0]
        self.feedforward_activations += (torch.abs(feedforward_activations) > self.activation_threshold).sum(dim=0).to("cpu").numpy()
        self.hidden_frequencies += (torch.abs(hidden_activations) > self.activation_threshold).sum(dim=0).to("cpu").numpy()
        top10_batch_activations, top10_batch_indices = torch.topk(hidden_activations, 10, dim=0)

        for i in range(self.dim_rala):

            new_neuron_activations = top10_batch_activations[:, i].to("cpu")
            new_neuron_tokens = batch_tokens[top10_batch_indices[:, i].tolist()]
            new_neuron_contexts = batch_contexts[(top10_batch_indices[:, i] // context_length).tolist()]

            self.neurons[i].update_neuron(
                new_neuron_activations, new_neuron_tokens, new_neuron_contexts
            )

        

    def save_to_files(self, folder_path, to_mlflow=False):
        """
        For mlflow=True, it should be run in a mlflow run
        """
        if self.n_activations == 0:
            raise ValueError("No activations to save")
        
        # Reset folder
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            try:
                shutil.rmtree(folder_path)
            except FileNotFoundError:
                print(f"The folder '{folder_path}' does not exist.")
            os.makedirs(folder_path)

        hidden_frequencies_file_name = os.path.join(folder_path, "hidden_frequencies.csv")
        feedforward_frequencies_file_name = os.path.join(folder_path, "feedforward_frequencies.csv")
        hidden_df = pd.DataFrame({"frequency": self.hidden_frequencies / self.n_activations})
        feedforward_df = pd.DataFrame({"frequency": self.feedforward_activations / self.n_activations})
        hidden_df.to_csv(hidden_frequencies_file_name, index=False)
        feedforward_df.to_csv(feedforward_frequencies_file_name, index=False)
        
        for neuron in self.neurons:
            neuron.save_to_csv(folder_path)

        if to_mlflow:
            os.system(f"zip -r {folder_path}.zip {folder_path}")
            mlflow.log_artifact(local_path=f"{folder_path}.zip")
            os.system(f"rm {folder_path}.zip")
