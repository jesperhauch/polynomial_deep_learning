import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch.distributions as dist
from typing import Callable, List

class PolynomialGenerator(pl.LightningDataModule): # TODO: Rewrite to have sample generation fn argument instead of distribution. 
    def __init__(self, polynomial: Callable[[int], float],  data_len: int, n_features: int, distribution = dist.normal.Normal(0,1), noise = None):
        self.poly_fn = polynomial
        self.n_data = data_len
        self.n_features = n_features
        self.data_dist = distribution
        self.noise = noise
        self.means = [-50, 0, 90]
        self.stds = [1, 5, 25]
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = False
        self.save_hyperparameters()

    def prepare_data(self):
        # do nothing
        pass

    def setup(self, stage: str):
        if stage == "fit":
            feature_samples = self.data_dist.sample((self.n_data, self.n_features))
            target_samples = self.poly_fn(*feature_samples.clone().permute(1,0))
            if self.noise:
                target_samples = target_samples + self.noise.sample((self.n_data,))
            train_set, val_set = random_split(feature_samples, [0.8, 0.2])
            self.train_set = TensorDataset(feature_samples[train_set.indices], target_samples[train_set.indices])
            self.val_set = TensorDataset(feature_samples[val_set.indices], target_samples[val_set.indices])
        
        if stage == "test":
            test_sets = []
            for mu in self.means:
                for std in self.stds:
                    feature_samples = dist.normal.Normal(mu, std).sample((self.n_data, self.n_features))
                    target_samples = self.poly_fn(*feature_samples.clone().permute(1,0))
                    if self.noise:
                        target_samples = target_samples + self.noise.sample((self.n_data,))
                    test_sets.append(TensorDataset(feature_samples, target_samples))
            self.test_sets = test_sets
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=32, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=32, num_workers=0)
    
    def test_dataloader(self) -> List[DataLoader]:
        return [DataLoader(dataset, batch_size=32, num_workers=0) for dataset in self.test_sets]

if __name__ == "__main__":
    poly = lambda a, b, c, d: (2*a**3)*(b**2)*(c**2)*(d**3)
    data = PolynomialGenerator(poly, 50000, poly.__code__.co_argcount, dist.normal.Normal(0,5))
    data.setup("fit")
    train = data.train_dataloader()
