import torch
from typing import List

class StandardScaler:
    """From https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40
    """
    def __init__(self, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
    
def shuffle_2d_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Shuffles independently along the second dimension of 2D tensor.

    Args:
        tensor (torch.Tensor): Matrix of shape (N,D) where shuffling is done along N dimension for each D.

    Returns:
        torch.Tensor: Shuffled tensor
    """
    shuffle_idxs = torch.ones_like(tensor, dtype=torch.int64)
    for i in range(tensor.shape[1]):
        torch.randperm(len(tensor), out=shuffle_idxs[:,i])
    return tensor.gather(0, shuffle_idxs)