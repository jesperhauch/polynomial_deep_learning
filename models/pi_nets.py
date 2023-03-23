import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from .neural_nets import FeedForwardNN, PolynomialActivation

class CCP(FeedForwardNN):
    def __init__(self, input_size=2, hidden_size=16, n_degree=4, output_size=1, bias=True):
        """
        This is the initialization function of the network, which in this case is a polynomial network.
        The implementation here relies on the CCP model of $\Pi$-nets. 
        https://github.com/polynomial-nets/tutorial-2022-intro-polynomial-nets/blob/master/pytorch/Minimum_example_pytorch.ipynb
        """
        super(FeedForwardNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_degree = n_degree
        for i in range(1, self.n_degree + 1):
            setattr(self, 'U{}'.format(i), nn.Linear(self.input_size, self.hidden_size, bias=bias))
        self.C = nn.Linear(self.hidden_size, self.output_size, bias=True)
        self.loss = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, z):
        out = self.U1(z)
        for i in range(2, self.n_degree + 1):
            out = getattr(self, 'U{}'.format(i))(z) * out + out
        out = self.C(out)
        return out.squeeze()

class CCP_relu(CCP):
    def forward(self,z):
        out = F.relu(self.U1(z))
        for i in range(2, self.n_degree + 1):
            out = F.relu(getattr(self, 'U{}'.format(i))(z)) * out + out
        out = self.C(out)
        return out.squeeze()

class CCP_relu_poly(CCP):
    def __init__(self, input_size=2, hidden_size=16, n_degree=4, output_size=1, bias=True):
        super().__init__(input_size, hidden_size, n_degree, output_size)
        self.poly_transfer = PolynomialActivation(n_degree)

    def forward(self, z):
        out = F.relu(self.U1(z))
        for i in range(2, self.n_degree + 1):
            out = F.relu(getattr(self, 'U{}'.format(i))(z)) * out + out
        out = self.poly_transfer(out)
        out = self.C(out)
        return out.squeeze()
    
if __name__ == "__main__":
    model = CCP(1, 16, 2, 1)
    data = torch.randn((10000, 1))
    model.forward(data)