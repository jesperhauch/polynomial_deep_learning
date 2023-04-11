import torch.nn as nn
import pytorch_lightning as pl
from typing import List
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


class BaseModel(pl.LightningModule):
    loss = nn.MSELoss()
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
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
        self.log('val_rmse', mean_squared_error(y, y_hat, squared=False), on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        self.log('test_r2', r2_score(y, y_hat))
        self.log('test_mae', mean_absolute_error(y, y_hat))
        self.log("test_mape", mean_absolute_percentage_error(y, y_hat))
        self.log("test_rmse", mean_squared_error(y, y_hat, squared=False))

class BaseTemporalModel(pl.LightningModule):
    loss = nn.MSELoss()
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def init_hidden(self):
        hidden = torch.zeros(self.n_layers, self.hidden_size)
        return hidden