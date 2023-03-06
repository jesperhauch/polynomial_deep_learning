import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class CCP(pl.LightningModule):
    def __init__(self, input_size=2, hidden_size=16, n_degree=4, output_size=1, bias=True):
        """
        This is the initialization function of the network, which in this case is a polynomial network.
        The implementation here relies on the CCP model of $\Pi$-nets. 
        https://github.com/polynomial-nets/tutorial-2022-intro-polynomial-nets/blob/master/pytorch/Minimum_example_pytorch.ipynb
        """
        super(CCP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_degree = n_degree
        for i in range(1, self.n_degree + 1):
            setattr(self, 'U{}'.format(i), nn.Linear(self.input_size, self.hidden_size, bias=bias))
        self.C = nn.Linear(self.hidden_size, self.output_size, bias=True)
        self.save_hyperparameters()

    def forward(self, z):
        out = self.U1(z)
        for i in range(2, self.n_degree + 1):
            out = getattr(self, 'U{}'.format(i))(z) * out + out
        out = self.C(out)
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        return loss


if __name__ == "__main__":
    data = torch.randn((64,25,2))
    y = torch.randn((64,25))
    model = CCP(2, 10, 4, 1)
    model.forward(data)
    model.training_step((data, y), 0)