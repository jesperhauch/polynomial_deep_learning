import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel

class CCP(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, n_degree: int, output_size: int, bias: bool=True, **kwargs):
        """
        This is the initialization function of the network, which in this case is a polynomial network.
        The implementation here relies on the CCP model of $\Pi$-nets. 
        https://github.com/polynomial-nets/tutorial-2022-intro-polynomial-nets/blob/master/pytorch/Minimum_example_pytorch.ipynb
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_degree = n_degree
        for i in range(1, self.n_degree + 1):
            setattr(self, 'U{}'.format(i), nn.Linear(self.input_size, self.hidden_size, bias=bias))
        self.C = nn.Linear(self.hidden_size, self.output_size, bias=True)

    def forward(self, z):
        out = self.U1(z)
        for i in range(2, self.n_degree + 1):
            out = getattr(self, 'U{}'.format(i))(z) * out + out
        out = self.C(out)
        return out.squeeze()

class PDC(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, n_degree: int, output_size: int, bias: bool=True, **kwargs):
        """
        This is the initialization function of the network, which in this case is a polynomial network.
        The implementation here relies on the PDC. 
        https://github.com/grigorisg9gr/polynomials-for-augmenting-NNs/blob/master/pdc_nosharing.py
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_degree = n_degree
        for i in range(1, self.n_degree + 1):
            setattr(self, 'V{}'.format(i), nn.Linear(self.hidden_size, self.hidden_size, bias=bias))
            for j in range(1, i+1):
                setattr(self, 'U{}_{}'.format(i, j), nn.Linear(self.input_size, self.hidden_size, bias=bias))
        self.C = nn.Linear(self.hidden_size, self.output_size, bias=True)

    def forward(self, z):
        out = self.U1_1(z)
        for i in range(2, self.n_degree + 1):
            out_deg = torch.ones_like(out)
            for j in range(1, i+1):
                out_deg *= getattr(self, 'U{}_{}'.format(i,j))(z)
            out = getattr(self, 'V{}'.format(i))(out_deg) + out
        out = self.C(out)
        return out.squeeze()
    
class PDCLow(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, n_degree: int, output_size: int, bias: bool=True, **kwargs):
        """
        This is the initialization function of the network, which in this case is a polynomial network.
        The implementation here relies on the PDC. 
        https://github.com/grigorisg9gr/polynomials-for-augmenting-NNs/blob/master/pdc_nosharing.py
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_degree = n_degree
        for i in range(1, self.n_degree + 1):
            for j in range(1, i+1):
                setattr(self, 'U{}_{}'.format(i, j), nn.Linear(self.input_size, self.hidden_size, bias=bias))
        self.C = nn.Linear(self.hidden_size, self.output_size, bias=True)

    def forward(self, z):
        out = self.U1_1(z)
        for i in range(2, self.n_degree + 1):
            out_deg = torch.ones_like(out)
            for j in range(1, i+1):
                out_deg *= getattr(self, 'U{}_{}'.format(i,j))(z)
            out += out_deg
        out = self.C(out)
        return out.squeeze()