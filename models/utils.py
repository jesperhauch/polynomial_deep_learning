import torch.nn as nn
import torch

class PolynomialActivation(nn.Module):
    def __init__(self, pow: int = 2):
        super().__init__()
        self.power = pow

    def forward(self, x):
        return torch.pow(x, self.power)
    
class WeightedMSELoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, input, target) -> torch.Tensor:
        weights = target.mean(axis=0)
        weights = 1 - self.softmax(weights)
        return torch.mean(weights * (input - target) ** 2)
    

class RootRelativeSquaredError(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, y_hat, y):
        numerator = torch.sum((y_hat-y)**2)
        denominator = torch.sum((y-y.mean())**2)
        return torch.sqrt(numerator/denominator)
    
