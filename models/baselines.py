import torch
import torch.nn as nn
from sklearn.preprocessing import PolynomialFeatures

class LinearRegression(nn.Module):
    def __init__(self, input_size: int, output_size: int, **kwargs):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)