import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

import numpy as np
import pandas as pd
import os
import sys

sys.path.append("../..")
from modelos.autoencoder.autoencoder_mio import Autoencoder as auto
from modelos.GPT.text_loader import TextLoader as textLoader


class Activations2:
    def __init__(self, autoencoder: auto, textloader: textLoader, feature_id: int):
        
        self.feature_id = feature_id
        self.autoencoder = autoencoder
        self.textloader = textloader

        self.batch_activations = torch.empty(0)
        self.batch_tokens = np.array([""] * 32)
        self.batch_contexts = np.array([[""]] * 32)

    def update_textLoader(self, new_textLoader : textLoader):
        self.textloader = new_textLoader
        return

    def get_batch_tokens(self):
        x, y = self.textloader.get_batch()
        
        indexs = y[:, -1]
        
        return indexs

    def get_batch_context(self):
        x, y = self.textloader.get_batch()

        return x

    def get_activations_batch(self, autoencoder_input):
        
        self.autoencoder.eval()
        hidden_activations = []

        def hook_fn(module, input, output):
            hidden_activations.append(output.detach().numpy()) 

        hook = self.autoencoder.relu1.register_forward_hook(hook_fn)

        for example in autoencoder_input:
            prediction = self.autoencoder(example)

        hook.remove()

        all_hidden_activations = np.vstack(hidden_activations)

        return all_hidden_activations


    def update_top10_with_batch(self):
 
        context = self.get_batch_context()
        tokens = self.get_batch_tokens()

        activations = self.get_activations_batch(context)

        top10_values, top10_index = torch.topk(activations[:, self.feature_id], 10)

        top_context = context[top10_index]
        top_indexs = tokens[top10_index]

        top20_activations = torch.cat((self.top10_activations, top10_values))
        top20_context = torch.cat((self.top10_context, top_context))
        top20_indexs = torch.cat((self.top10_indexs, top_indexs))

        new_top10_values, new_top10_indices = torch.topk(top20_activations, 10)

        self.top10_activations = new_top10_values
        self.top10_context = top20_context[new_top10_indices]
        self.top10_indexs = top20_indexs[new_top10_indices]


class Neuron:
    def __init__(self, feature_id:int):
        self.activations = torch.empty(10)
        self.tokens = np.array([""] * 10)
        self.contexts = np.array([[""]] * 10)

        self.feature_id = feature_id


    def update_neuron(self, new_activations:torch.tensor, new_tokens:torch.tensor, new_contexts:torch.tensor):
        
        top_activations = torch.cat((self.activations, new_activations))
        top_tokens = torch.cat((self.tokens, new_tokens))
        top_contexts = torch.cat((self.contexts, new_contexts))

        new_top10_activations, new_top10_indexs = torch.topk(top_activations, 10)

        self.activations = new_top10_activations
        self.tokens = top_tokens[new_top10_indexs]
        self.contexts = top_contexts[new_top10_indexs]

        return
    
    def get_data(self):
        return self.activations, self.tokens, self.contexts
    
    def create_csv(self, folder_path):
        file_name = f'activations_feature_{self.feature_id}.csv'

        #folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        final_path = os.path.join(folder_path, file_name)
        
        data = {
            'Activacion': self.activations,
            'Tokens Id': self.tokens,
            'Context': self.contexts
        }
        
        df = pd.DataFrame(data)

        df.to_csv(final_path, index=False)
        return

class Activations:
    def __init__(self, autoencoder: auto, textloader: textLoader, batch_size:int, dim_rala:int):
        
        self.batch_size = batch_size
        self.dim_rala = dim_rala

        self.neurons = [Neuron(i) for i in range(dim_rala)]
        self.autoencoder = autoencoder
        self.textloader = textloader


    def get_batch_tokens(self):
        x, y = self.textloader.get_batch()
        
        indexs = y[:, -1]
    
        return indexs

    def get_batch_context(self):
        x, y = self.textloader.get_batch()

        return x

    def update_batch_data(self, autoencoder_input):
        #autoencoder_input tiene un formato batch_size x emb_size_post_transformers (32 x 128)
        #post autoencoder queda algo de batch_size x emb_size_dim_rala (32 x 1024)
        
        batch_tokens = self.get_batch_tokens()
        batch_contexts = self.get_batch_context()

        self.autoencoder.eval()
        hidden_activations = []

        def hook_fn(module, input, output):
            hidden_activations.append(output.detach().numpy()) 

        hook = self.autoencoder.relu1.register_forward_hook(hook_fn)

        for example in autoencoder_input:
            prediction = self.autoencoder(example)

        hook.remove()

        all_hidden_activations = torch.tensor(np.vstack(hidden_activations))

        top10_batch_activations, top10_batch_indices = torch.topk(all_hidden_activations, 10, dim=0)

        for i in range(self.dim_rala):

            new_neuron_activations = top10_batch_activations[:, i]
            new_neuron_tokens = batch_tokens[top10_batch_indices[:, i]]
            new_neuron_contexts = batch_contexts[top10_batch_indices[:, i]]

            self.neurons[i].update_neuron(new_neuron_activations, new_neuron_tokens, new_neuron_contexts)

        return all_hidden_activations

    def get_data_neuron(self, neuron_id:int):
        self.neurons[neuron_id].get_data()
        return

    def create_csv_neuron(self, neuron_id:int, folder_path):
        self.neurons[neuron_id].create_csv(folder_path)
        return


#como estas facundo
#estas facundo bien
