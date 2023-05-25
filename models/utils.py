import torch.nn as nn
import torch

class PolynomialActivation(nn.Module):
    """Class for implementing polynomial activation for the PANN model.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, pow: int = 2):
        super().__init__()
        self.power = pow

    def forward(self, x):
        return torch.pow(x, self.power)
    
class RootRelativeSquaredError(nn.Module):
    """Root relative squared error metric. Defined as https://www.gepsoft.com/gxpt4kb/Chapter10/Section1/SS07.htm

    Args:
        nn (_type_): _description_
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, y_hat, y):
        numerator = torch.sum((y_hat-y)**2)
        denominator = torch.sum((y-y.mean())**2)
        return torch.sqrt(numerator/denominator)
    
