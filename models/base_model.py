import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MultioutputWrapper, R2Score, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError 
from typing import List, Tuple
from models.utils import MaxMin, WeightedMSELoss
from torch.nn import MSELoss
from models.utils import WeightedMSELoss

class BaseModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        self.save_hyperparameters()
        self.r2 = R2Score()
        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()
        self.mse = MeanSquaredError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
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
        self.log('val_r2', self.r2(y_hat, y), on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        self.log('test_r2', self.r2(y_hat, y))
        self.log('test_mae', self.mae(y_hat, y))
        self.log("test_mape", self.mape(y_hat, y))
        self.log("test_mse", self.mse(y_hat, y))

class SIRModelWrapper(BaseModel):
    log_features = ["S", "I", "R"]
    val_metrics = ["r2", "mape", "mse"]
    test_metrics = ["r2", "mape", "mae", "mse"]
    def __init__(self, multiplication_net: BaseModel, input_size: int, hidden_size: int, n_degree: int, scale: bool=False, loss_fn: str = "MSELoss", **kwargs):
        super().__init__()
        self.r2 = MultioutputWrapper(R2Score(), num_outputs=3)
        self.mae = MultioutputWrapper(MeanAbsoluteError(), num_outputs=3)
        self.mape = MultioutputWrapper(MeanAbsolutePercentageError(), num_outputs=3)
        self.mse = MultioutputWrapper(MeanSquaredError(), num_outputs=3)
        self.scale = scale
        self.scaler = MaxMin()
        try:
            self.loss = eval(loss_fn, globals())()
        except Exception as inst:
            print(inst)
            raise NotImplementedError("The loss function {n} is not implemented or imported correctly.")
          
        self.s_net = multiplication_net(input_size, hidden_size, n_degree, 1)
        self.i_net = multiplication_net(input_size, hidden_size, n_degree, 1)
        self.r_net = multiplication_net(input_size, hidden_size, n_degree, 1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.save_hyperparameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.s_net(x)
        i = self.i_net(x)
        r = self.r_net(x)
        output =  torch.stack((s,i,r), dim=-1)
        return self.softmax(output)

    def training_step(self, batch, batch_idx):
        beta, gamma, X, y = batch
        beta = beta.unsqueeze(-1).repeat(1,X.size(1)).unsqueeze(-1).to(torch.float32)
        gamma = gamma.unsqueeze(-1).repeat(1,X.size(1)).unsqueeze(-1).to(torch.float32)
        if self.scale:
            X = self.scaler.transform(X)
            y = self.scaler.transform(y)
        X_input = torch.concat([X, beta, gamma], dim=-1)
        y_hat = self(X_input).reshape(y.shape)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        beta, gamma, X, y = batch
        beta = beta.reshape(len(X), 1).to(torch.float32)
        gamma = gamma.reshape(len(X), 1).to(torch.float32)
        if self.scale:
            X = self.scaler.transform(X)
            y = self.scaler.transform(y)
        y_hat = torch.zeros_like(y)
        X_forward = torch.concat([X[:,0,:], beta, gamma], dim=1).unsqueeze(1) # Only use first observation
        for t in range(X.size(1)):
            next_state = self(X_forward)
            y_hat[:, t, :] = next_state
            X_forward = torch.concat([next_state, beta, gamma], dim=1).unsqueeze(1)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)

        # Extra logging metrics - need to flatten batch dimension
        y = y.flatten(0,1)
        y_hat = y_hat.flatten(0,1)
        r2 = self.r2(y_hat, y)
        mape = self.mape(y_hat, y)
        mse = self.mse(y_hat, y)
        values = [r2, mape, mse]
        self.log_dict({f"val_{metric}_{feat}": values[i][j] for i, metric in enumerate(self.val_metrics) for j, feat in enumerate(self.log_features)})
        return loss
    
    def test_step(self, batch, batch_idx):
        beta, gamma, X, y = batch
        beta = beta.reshape(len(X), 1).to(torch.float32)
        gamma = gamma.reshape(len(X), 1).to(torch.float32)
        if self.scale:
            X = self.scaler.transform(X)
            y = self.scaler.transform(y)
        y_hat = torch.zeros_like(y)
        X_forward = torch.concat([X[:,0,:], beta, gamma], dim=1).unsqueeze(1) # Only use first observation
        for t in range(X.size(1)):
            next_state = self(X_forward)
            y_hat[:, t, :] = next_state
            X_forward = torch.concat([next_state, beta, gamma], dim=1).unsqueeze(1)

        if self.scale:
            y = self.scaler.untransform(y)
            y_hat = self.scaler.untransform(y_hat)

        # Extra logging metrics - need to flatten batch dimension
        y = y.flatten(0,1)
        y_hat = y_hat.flatten(0,1)
        r2 = self.r2(y_hat, y)
        mape = self.mape(y_hat, y)
        mae = self.mae(y_hat, y)
        mse = self.mse(y_hat, y)
        values = [r2, mape, mae, mse]
        self.log_dict({f"test_{metric}_{feat}": values[i][j] for i, metric in enumerate(self.test_metrics) for j, feat in enumerate(self.log_features)})
    
    @torch.no_grad()
    def generate_sequence(self, N: int, X_start: torch.Tensor, beta: float, gamma: float):
        beta = torch.Tensor([beta])
        gamma = torch.Tensor([gamma])
        X_forward = torch.concat([X_start, beta, gamma], dim=-1)
        simulation = []
        for _ in range(N):
            next_state = self(X_forward)
            simulation.append(next_state)
            X_forward = torch.concat([next_state, beta, gamma], dim=-1)
        
        return torch.stack(simulation)
    
if __name__ == "__main__":
    from models.pi_nets import CCP
    from torch.nn.functional import softmax
    model = SIRModelWrapper(CCP, 5, 64, 2)
    data = torch.randn(3)
    data = softmax(data, dim=0)
    model.generate_step(5, data, 0.05, 0.017)