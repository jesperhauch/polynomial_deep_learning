import torch
import torch.distributions as dist
from torch.utils.data import TensorDataset
from typing import Tuple, List
from data.utils import StandardScaler, shuffle_2d_tensor

class BaseDataClass:
    noise_dist = dist.normal.Normal(0,1)
    train_quantiles = torch.tensor([0.05, 0.95]) # assumes 90/10 train-test split
    def __init__(self, n_data: int, noise: bool = False, standardize: bool = False):
        # Train test split
        self.train_test_split(n_data)
        # Build features and sort ascending
        features_sorted, _ = self.build_features(n_data).sort(axis=0)
        # Train test split
        train_features = features_sorted[self.train_indices,:]
        test_features = features_sorted[self.test_indices,:]
        # Randomly shuffle features
        self.train_features = shuffle_2d_tensor(train_features)
        self.test_features = shuffle_2d_tensor(test_features)
        # Calculate targets
        self.train_targets = self.calculate_targets(*self.train_features.permute(1,0))
        self.test_targets = self.calculate_targets(*self.test_features.permute(1,0))

        # Add noise and standardize if necessary
        if noise:
            self.train_features = self.train_features + self.noise_dist.sample(self.train_features.shape)
            self.test_features = self.test_features + self.noise_dist.sample(self.test_features.shape)
            self.train_targets = self.train_targets + self.noise_dist.sample(self.train_targets.shape)
            self.test_targets = self.test_targets + self.noise_dist.sample(self.test_targets.shape)
        if standardize:
            self.standardize_features()

    def build_features(self, n_data: int) -> torch.Tensor:
        pass

    def calculate_targets(self, **kwargs) -> torch.Tensor:
        pass
    
    def standardize_features(self):
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        self.train_features = feature_scaler.fit_transform(self.train_features)
        self.test_features = feature_scaler.transform(self.test_features)
        self.train_targets = target_scaler.fit_transform(self.train_targets)
        self.test_targets = target_scaler.transform(self.test_targets)

    def n_features(self):
        assert self.train_features.shape[1] == self.test_features.shape[1]
        return self.train_features.shape[1]

    def train_test_split(self, n_data):
        self.train_indices = torch.arange(*(self.train_quantiles*n_data)).to(torch.int64)
        self.test_indices = torch.cat([torch.arange(0, self.train_indices[0]), torch.arange(self.train_indices[-1]+1, n_data)]).to(torch.int64)

    def generate_train_data(self) -> Tuple:
        return self.train_features, self.train_targets
    
    def generate_test_data(self) -> List[TensorDataset]:
        return [TensorDataset(self.test_features, self.test_targets)]