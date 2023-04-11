import torch
import torch.nn as nn
from models.base_model import BaseTemporalModel
from models.pi_nets import CCP
from typing import Tuple
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

class RNN(BaseTemporalModel):
    def __init__(self, input_size: int, hidden_size: int, n_degree: int, output_size: int, n_layers: int = 1, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size = input_size, 
                          hidden_size = hidden_size, 
                          num_layers = n_layers, 
                          batch_first=True)
        self.multi_net = CCP(hidden_size, hidden_size, n_degree, output_size)   
    
    def forward(self, x: torch.Tensor, hidden: torch.tensor) -> Tuple[torch.Tensor]:
        x = x.reshape(-1,1)
        output, hidden = self.rnn(x, hidden)
        output = self.multi_net(output)
        return output, hidden
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        loss = 0
        hidden = self.init_hidden()
        for i in range(len(X)):
            output, hidden = self(X[i], hidden)
            loss += self.loss(output, y[i])
        loss /= i+1
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        hidden = self.init_hidden()
        y_hat, _ = self.forward(X, hidden)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_r2', r2_score(y, y_hat), on_epoch=True)
        self.log('val_rmse', mean_squared_error(y, y_hat, squared=False), on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        hidden = self.init_hidden()
        y_hat, _ = self.forward(X, hidden)
        self.log('test_r2', r2_score(y, y_hat))
        self.log('test_mae', mean_absolute_error(y, y_hat))
        self.log("test_mape", mean_absolute_percentage_error(y, y_hat))
        self.log("test_rmse", mean_squared_error(y, y_hat, squared=False))
    

if __name__ == "__main__":
    from data.epidemiology import SimulationModule, EpidemiologyDataset
    ds = EpidemiologyDataset(1/80, 1/160, 1000, [], None)
    data_module = SimulationModule(ds, 32)
    data_module.setup("fit")
    train = data_module.train_dataloader()
    X,y = next(iter(train))
    model = RNN(1, 64, 3, 1)
    #hidden = model.init_hidden(len(X))
    #model.forward(X, hidden)
    model.training_step((X,y), 0)