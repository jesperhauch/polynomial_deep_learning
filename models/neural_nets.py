import torch.nn as nn
import pytorch_lightning as pl
import torch
from typing import List

class SquareActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.square(x)

class FeedForwardNN(pl.LightningModule):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super(FeedForwardNN, self).__init__()
        modules = [nn.Linear(input_size, hidden_sizes[0])]
        for i in range(len(hidden_sizes[:-1])):
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        modules.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*modules)
        self.loss = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.layers(x)
    
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
        return loss

class PolynomialNN(FeedForwardNN):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super(FeedForwardNN, self).__init__()
        modules = [nn.Linear(input_size, hidden_sizes[0])]
        for i in range(len(hidden_sizes[:-1])):
            modules.append(SquareActivation())
            modules.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        modules.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*modules)
        self.loss = nn.MSELoss()
        self.save_hyperparameters()

class RNN(pl.LightningModule):
    def __init__(self, input_size, hidden_size, n_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, n_layers)
        self.out_fc = nn.Linear(hidden_size, output_size)
        self.save_hyperparameters()

    def forward(self, x):
        out, h = self.rnn(x)
        out = self.out_fc(out)
        out = torch.sigmoid(out)
        return out.squeeze(-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

if __name__ == "__main__":
    model = PolynomialNN(1, [500, 250, 125], 1)
    test = SquareActivation()
    data = torch.randn((32,1))
    model(data)