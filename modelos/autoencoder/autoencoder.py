import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


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

        # Este es el pre encoder bias que muestran en las cuentas del autoencoder antes de entrar en el encoder.
        # Entiendo que es un bias distinto al de la capa encoder porque ese va después de aplicar la matriz de pesos.
        self.pre_encoder_bias = nn.Parameter(torch.Tensor())

    def encode(self, x):
        x = x - self.pre_encoder_bias
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x) + self.pre_encoder_bias


class lossAutoencoder(nn.Module):
    def __init__(self, autoencoder: Autoencoder, lasso_lambda: int = 1e-3):
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
    # Preparamos el dataset:
    # Hacemos un dataset customizado de activaciones de neuronas suponiendo que activaciones
    # es un numpy array.
    class DatasetActivaciones(Dataset):
        def __init__(self, activaciones):
            self.activaciones = [torch.tensor(lista_activaciones, dtype=torch.float32) for lista_activaciones in activaciones]

        def __len__(self):
            return len(self.activaciones)

        def __getitem__(self, idx):
            activation = self.activaciones[idx]
            return activation  # Sin labels, solo un valor

    # Creamos un loader iterable indicandole que debe leer los datos a partir de
    # del dataset creado en el paso anterior. Este objeto puede ser iterado
    # y nos devuelve de a un batch (x, y).
    dataset = DatasetActivaciones(torch.from_numpy(activaciones).clone())

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


    #Definimos el optimizador
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    #Definimos la función de pérdida
    loss_function = lossAutoencoder(autoencoder, lasso_lambda)

    #Entrenamiento
    for epoch in range(epochs):

        total_loss= .0

        for x_train in loader:
            optimizer.zero_grad()

            '''
            ¿Cómo hago para computar el forward, la loss y después el backward? Si computo la loss,
            ¿cuenta como haber pasado forward?
            '''
            # Realizo la pasada forward por la red
            loss = loss_function(autoencoder)

            # Realizo la pasada backward por la red
            loss.backward()

            # Actualizo los pesos de la red con el optimizador
            optimizer.step()

            # Me guardo el valor actual de la función de pérdida para luego graficarlo
            loss_list.append(loss.data.item())

            # Acumulo la loss del minibatch
            total_loss += loss.item() * y.size(0)

            # Normalizo la loss total
        total_loss /= len(loader.dataset)

        # Muestro el valor de la función de pérdida cada 100 iteraciones
        if i > 0 and i % 100 == 0:
            print('Epoch %d, loss = %g' % (i, total_loss))

    # Muestro la lista que contiene los valores de la función de pérdida
    # y una versión suavizada (rojo) para observar la tendencia
    plt.figure()
    loss_np_array = np.array(loss_list)
    plt.plot(loss_np_array, alpha=0.3)
    N = 60
    running_avg_loss = np.convolve(loss_np_array, np.ones((N,)) / N, mode='valid')
    plt.plot(running_avg_loss, color='red')
    plt.title("Función de pérdida durante el entrenamiento")