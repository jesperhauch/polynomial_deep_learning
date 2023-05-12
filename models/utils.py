import torch.nn as nn
import torch

class MaxMin():
    """MinMax scaler class
    """
    def __init__(self) -> None:
        self.eps = 1e-5

    def transform(self, data):
        min_values = data.min(axis=0).values
        max_values = data.max(axis=0).values
        self.params = {"min_values": min_values,
                       "max_values": max_values}
        tf_data = (data.clone() - min_values)/(max_values + self.eps)
        return tf_data

    def untransform(self, data):
        rtf_data = data.clone()*self.params['max_values'] + self.params['min_values']
        return rtf_data
    
class WeightedMSELoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, input, target) -> torch.Tensor:
        weights = target.mean(axis=0)
        weights = 1 - self.softmax(weights)
        return torch.mean(weights * (input - target) ** 2)