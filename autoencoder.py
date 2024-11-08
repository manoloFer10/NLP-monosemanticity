import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from torchinfo import summary


class Autoencoder(nn.Module):
    def __init__(self, dim_activaciones: int, dim_rala: int, dataset_geometric_median, device):
        super().__init__()

        self.encoder = nn.Linear(dim_activaciones, dim_rala)
        self.decoder = nn.Linear(dim_rala, dim_activaciones)
        self.relu = nn.ReLU()
    
        self.dim_activaciones = dim_activaciones
        self.dim_rala = dim_rala
        self.device = device

        # Este es el pre encoder bias que muestran en las cuentas del autoencoder antes de entrar en el encoder.
        # Entiendo que es un bias distinto al de la capa encoder porque ese va después de aplicar la matriz de pesos.
        # Inicializamos como la mediana geométrica del dataset (lo dice en el paper de anthropic).
        # OJETE: la mediana geometrica debería ser sobre las activaciones de las neuronas. O sea, habría que precomputarlas
        # Muy costoso. Ver qué hacemos con esto.

        self.pre_encoder_bias = self.decoder.bias

    def encode(self, x):
        x = x - self.pre_encoder_bias
        x = self.encoder(x)
        x = self.relu(x)

        return x

    def decode(self, x):
        x = self.decoder(x)
        x = self.relu(x)
        x = self.pre_encoder_bias + x

        return x

    def forward(self, x):

        encoded = self.encode(x)
        decoded = self.decode(encoded)

        return encoded, decoded

    def normalize_decoder_weights(self):
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, p=2, dim=1)

    @classmethod
    def load_from_mlflow(cls, experiment, run_id, device="cpu"):
        mlflow.set_experiment(experiment)
        local_model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="autoencoder/data/model.pth"
        )
        model = torch.load(local_model_path, map_location=device)
        model.device = device
        model.to(device)
        return model

class LossAutoencoder(nn.Module):
    def __init__(self, lasso_lambda: int = 1e-3):
        super(LossAutoencoder, self).__init__()
        self.lasso_lambda = lasso_lambda

    def lasso_loss(self, f):
        l1 = torch.norm(f, p=1)
        return self.lasso_lambda * l1

    def forward(self, input_activaciones, encoded, decoded):
        mse = nn.MSELoss()

        ecm = mse(decoded, input_activaciones)
        lasso = self.lasso_loss(encoded)

        return ecm + lasso


# TODO: terminar el .train() del autoencoder para usar en el bucle de entrenamiento con el transformer.
# Lo de abajo lo dejo para reciclar cuando tengamos que entrenar. Me fui al pasto.


def entrenar_autoencoder(
    autoencoder: Autoencoder,
    activaciones: Dataset,
    lasso_lambda: float,
    epochs: int,
    batch_size: int,
    learning_rate=1e-3,
):
    # Preparamos el dataset:
    # Hacemos un dataset customizado de activaciones de neuronas suponiendo que activaciones
    # es un numpy array.

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    class DatasetActivaciones(Dataset):
        def __init__(self, activaciones):
            self.activaciones = [
                torch.tensor(lista_activaciones, dtype=torch.float32)
                for lista_activaciones in activaciones
            ]

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

    # Definimos el optimizador
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Definimos la función de pérdida
    loss_function = LossAutoencoder(lasso_lambda)

    # Lista en la que iremos guardando el valor de la función de pérdida en cada
    # etapa de entrenamiento
    loss_list = []

    with mlflow.start_run():
        params = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": "Adam",
        }
        mlflow.log_params(params)

        with open("autoencoder_summary.txt", "w") as f:
            f.write(str(summary(autoencoder)))

        mlflow.log_artifact("autoencoder_summary.txt")

        # Entrenamiento
        for epoch in range(epochs):
            total_loss = 0.0

            for x_train in loader:
                input_data = x_train.to(device)
                optimizer.zero_grad()

                # Realizo la pasada forward por la red
                encoded, decoded = autoencoder(input_data)

                loss = loss_function(input_activaciones=x_train, encoded=encoded, decoded=decoded)

                # Realizo la pasada backward por la red
                loss.backward()

                # Actualizo los pesos de la red con el optimizador
                optimizer.step()

                # Me guardo el valor actual de la función de pérdida para luego graficarlo
                loss_list.append(loss.data.item())

                # Acumulo la loss del minibatch
                total_loss += loss.item() * x_train.size(0)

                # Normalizo la loss total
            total_loss /= len(loader.dataset)

            # Muestro el valor de la función de pérdida cada 100 iteraciones
            if epoch > 0 and epoch % 100 == 0:
                mlflow.log_metric(
                    "lasso_loss", f"{total_loss:.4f}", step=len(loader.dataset) * epochs
                )
                print("Epoch %d, loss = %g" % (epoch, total_loss))

        mlflow.pytorch.log_model(autoencoder, "autoencoder")
    # Muestro la lista que contiene los valores de la función de pérdida
    # y una versión suavizada (rojo) para observar la tendencia
    plt.figure()
    loss_np_array = np.array(loss_list)
    plt.plot(loss_np_array, alpha=0.3)
    N = 60
    running_avg_loss = np.convolve(loss_np_array, np.ones((N,)) / N, mode="valid")
    plt.plot(running_avg_loss, color="red")
    plt.title("Función de pérdida durante el entrenamiento")
