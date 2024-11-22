import torch
import torch.nn as nn
import mlflow

class Autoencoder(nn.Module):
    def __init__(self, dim_activaciones: int, dim_rala: int, device):
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

        # NOTE: Hice esto para no tener que calcular la mediana geometrica. Lo ponemos como un parametreo entrenable y adios.
        self.pre_encoder_bias = nn.Parameter(torch.zeros(dim_activaciones, device=device))

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

    def lasso_loss(self, encoded):
        # NOTE: Antes haciamos esto
        # l1 = torch.norm(f, p=1)
        
        # Pero estaba mal porque computabamos la norma de todo el batch
        # ahora computamos la norma de cada vector de activaciones y promediamos
        # Como que antes fomentabamos espacidad en todo el batch (creo que no tiene sentido)
        l1 = encoded.norm(p=1, dim=-1).mean()
        return self.lasso_lambda * l1.mean()

    def forward(self, input_activaciones, encoded, decoded):
        mse = nn.MSELoss()

        ecm = mse(decoded, input_activaciones)
        lasso = self.lasso_loss(encoded)

        return ecm + lasso