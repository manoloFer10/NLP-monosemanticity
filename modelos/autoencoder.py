import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset


class Autoencoder(nn.Module):
    def __init__(self, dim_activaciones: int, dim_rala: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(dim_activaciones, dim_rala),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_rala, dim_activaciones),
            nn.ReLU()
        )

        self.pre_encoder_bias = nn.Parameter(torch.Tensor())

    def encode(self, x):
        x = x - self.pre_encoder_bias
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x) + self.pre_encoder_bias


class lossAutoencoder(nn.Module):
    def __init__(self, autoencoder: Autoencoder, lasso_lambda: int):
        super(lossAutoencoder, self).__init__()
        self.autoencoder = autoencoder
        self.lasso_lambda = lasso_lambda

    def lasso_loss(self, f):
        l1 = torch.norm(f, p=1)
        return self.lasso_lambda * l1

    def forward(self, input_activaciones):
        mse = nn.MSELoss()

        f = self.autoencoder.encode(input_activaciones)
        reconstruccion = self.autoencoder.decode(f)

        ecm = mse(reconstruccion, input_activaciones)
        lasso = self.autoencoder.lasso_loss(f)

        return ecm + lasso

'''
PENDIENTE
'''
def entrenar_autoencoder(autoencoder: Autoencoder, activaciones:Dataset, lasso_lambda: int, epochs: int, batch_size: int, learning_rate=1e-3):
    #Preparamos el dataset:
    #Hacemos un dataset customizado de activaciones de neuronas suponiendo que activaciones
    class DatasetActivaciones(Dataset):
        def __init__(self, activaciones):
            self.activaciones = torch.tensor(activaciones, dtype=torch.float32)

        def __len__(self):
            return len(self.activaciones)

        def __getitem__(self, idx):
            activation = self.activaciones[idx]
            return activation  # No labels, only return the input

    # Creamos un loader iterable indicandole que debe leer los datos a partir de
    # del dataset creado en el paso anterior. Este objeto puede ser iterado
    # y nos devuelve de a un batch (x, y).
    dataset = DatasetActivaciones(activaciones)

    loader = DataLoader(dataset=activaciones, batch_size=batch_size, shuffle=True)


    #Definimos el optimizador
    optimizer = torch.optim.Adam(autoencoder.parameters())

    #Definimos la función de pérdida
    loss = lossAutoencoder(autoencoder, lasso_lambda)

    #Entrenamiento
    for epoch in range(epochs):
        for x_train in loader:
            ...