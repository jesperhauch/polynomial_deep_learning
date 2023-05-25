import torch.nn as nn
import pytorch_lightning as pl
import torch
from typing import List
from models.base_model import BaseModel
from models.utils import PolynomialActivation

class FFNN(BaseModel):
    """Feed forward neural network (FFNN). 

    Args:
        BaseModel (_type_): _description_
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, **kwargs):
        super().__init__()
        n_layers = [input_size] + hidden_sizes
        modules = []
        for i in range(len(n_layers)-1):
            modules.append(nn.Linear(n_layers[i], n_layers[i+1]))

        modules.append(nn.Linear(n_layers[-1], output_size))
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x).squeeze()

class PANN(FFNN):
    """Polynomial Activation Neural Network (PANN).

    Args:
        FFNN (_type_): _description_
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, n_degree: int = 2, **kwargs):
        super().__init__(input_size, hidden_sizes, output_size)
        modules = []
        modules.append(nn.Linear(input_size, hidden_sizes[0]))
        
        for i in range(len(hidden_sizes)-1):
            modules.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        modules.append(PolynomialActivation(n_degree))
        modules.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.layers = nn.Sequential(*modules)