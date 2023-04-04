import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch.distributions as dist
from typing import Callable, List, Tuple

class NormalGenerator:
    def __init__(self, polynomial: Callable[[int], float], n_features: int, n_data: int, mu: float = 0, std: float = 5, noise: bool = False):
        self.polynomial = polynomial
        self.n_features = n_features
        self.n_data = n_data
        self.data_dist = dist.normal.Normal(mu, std)
        self.mu = mu
        self.std = std
        self.noise = noise
        # Predefined attributes
        self.noise_dist = dist.normal.Normal(0,1)
        self.means = [-50, 0, 90]
        self.stds = [1, 5, 25]

    def generate_train_data(self) -> Tuple:
        feature_samples = self.data_dist.sample((self.n_data, self.n_features))
        target_samples = self.polynomial(*feature_samples.clone().permute(1,0))
        if self.noise:
            target_samples = target_samples + self.noise_dist.sample((self.n_data,))
        return feature_samples, target_samples
    
    def generate_test_data(self) -> List[TensorDataset]:
        test_sets = []
        for mu in self.means:
            for std in self.stds:
                test_dist = dist.normal.Normal(mu, std)
                feature_samples = test_dist.sample((self.n_data, self.n_features))
                target_samples = self.polynomial(*feature_samples.clone().permute(1,0))
                if self.noise:
                    target_samples = target_samples + self.noise_dist.sample((self.n_data,))
                test_sets.append(TensorDataset(feature_samples, target_samples))
        return test_sets


class PolynomialModule(pl.LightningDataModule): # TODO: Rewrite to have sample generation fn argument instead of distribution. 
    def __init__(self, fn_data):
        self.fn_data = fn_data
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = False
        self.save_hyperparameters()

    def prepare_data(self):
        # do nothing
        pass

    def setup(self, stage: str):
        if stage == "fit":
            feature_samples, target_samples = self.fn_data.generate_train_data()
            train_set, val_set = random_split(feature_samples, [0.8, 0.2])
            self.train_set = TensorDataset(feature_samples[train_set.indices], target_samples[train_set.indices])
            self.val_set = TensorDataset(feature_samples[val_set.indices], target_samples[val_set.indices])
        
        if stage == "test":
            self.test_sets = self.fn_data.generate_test_data()
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=32, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=32, num_workers=0)
    
    def test_dataloader(self) -> List[DataLoader]:
        return [DataLoader(dataset, batch_size=32, num_workers=0) for dataset in self.test_sets]

if __name__ == "__main__":
    from simulation_functions import ShortColumn, SulfurModel
    data_gen = ShortColumn(100000)
    data = PolynomialModule(data_gen)
    data.setup("fit")
    train = data.train_dataloader()
