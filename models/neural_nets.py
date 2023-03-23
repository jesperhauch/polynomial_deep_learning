import torch.nn as nn
import pytorch_lightning as pl
import torch
from typing import List
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

class PolynomialActivation(nn.Module):
    def __init__(self, pow: int = 2):
        super().__init__()
        self.power = pow

    def forward(self, x):
        return torch.pow(x, self.power)

class FeedForwardNN(pl.LightningModule):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super(FeedForwardNN, self).__init__()
        n_layers = [input_size] + hidden_sizes
        modules = []
        for i in range(len(n_layers)-1):
            modules.append(nn.Linear(n_layers[i], n_layers[i+1]))
            modules.append(nn.ReLU(inplace=True))

        modules.append(nn.Linear(n_layers[-1], output_size))
        self.layers = nn.Sequential(*modules)
        self.loss = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.layers(x).squeeze()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_r2', r2_score(y, y_hat), on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        self.log('test_r2', r2_score(y, y_hat))
        self.log('test_mae', mean_absolute_error(y, y_hat))
        self.log("test_mape", mean_absolute_percentage_error(y, y_hat))
        return 

class PolynomialNN(FeedForwardNN):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, n_pow: int = 2):
        super(FeedForwardNN, self).__init__()
        modules = []
        modules.append(nn.Linear(input_size, hidden_sizes[0]))
        
        for i in range(len(hidden_sizes)-1):
            modules.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        modules.append(PolynomialActivation(n_pow))
        modules.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.layers = nn.Sequential(*modules)
        self.loss = nn.MSELoss()
        self.save_hyperparameters()

class PolynomialNN_each(FeedForwardNN):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super(FeedForwardNN, self).__init__()
        n_layers = [input_size] + hidden_sizes
        modules = []
        for i in range(len(n_layers)-1):
            modules.append(nn.Linear(n_layers[i], n_layers[i+1]))
            modules.append(PolynomialActivation(pow=2))

        modules.append(nn.Linear(n_layers[-1], output_size))
        self.layers = nn.Sequential(*modules)
        self.loss = nn.MSELoss()
        self.save_hyperparameters()

class PolynomialNN_relu(FeedForwardNN):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, n_pow: int = 2):
        super(FeedForwardNN, self).__init__()
        modules = []
        modules.append(nn.Linear(input_size, hidden_sizes[0]))
        
        for i in range(len(hidden_sizes)-1):
            modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        modules.append(PolynomialActivation(n_pow))
        modules.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.layers = nn.Sequential(*modules)
        self.loss = nn.MSELoss()
        self.save_hyperparameters()

if __name__ == "__main__":
    model = PolynomialNN(1, [64], 1, 3)
    print(model)
    data = torch.randn((32,1))
    model(data)