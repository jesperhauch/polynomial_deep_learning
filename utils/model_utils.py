import torch

def number_of_features(n_degree: int) -> int:
    """Calculate dimension of matrix when using polynomial features of some degree for EpidemiologyModule

    Args:
        n_degree (int): Polynomial degree

    Returns:
        int: Number of features in matrix
    """
    features = 0
    for i in range(1, n_degree+1):
        features += i+1
    return features

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)