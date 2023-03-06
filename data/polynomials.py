import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.distributions.normal import Normal
from typing import Callable

class PolynomialGenerator(pl.LightningDataModule):
    def __init__(self, polynomial: Callable[[int], float],  data_len: int, distribution = Normal(0,1)):
        self.poly_fn = polynomial
        self.n_data = data_len
        self.data_dist = distribution
        self.prepare_data_per_node = True
        self._log_hyperparams = False

    def prepare_data(self):
        # do nothing
        pass

    def setup(self, stage: str):
        if stage == "fit":
            feature_samples = self.data_dist.sample((self.n_data,1))
            target_samples = feature_samples.clone().apply_(self.poly_fn) # Only works on CPU
            train_set, val_set = random_split(feature_samples, [0.8, 0.2])
            self.train_set = TensorDataset(feature_samples[train_set.indices], target_samples[train_set.indices])
            self.val_set = TensorDataset(feature_samples[val_set.indices], target_samples[val_set.indices])
        
        if stage == "test":
            feature_samples = self.data_dist.sample((self.n_data,1))
            target_samples = feature_samples.clone().apply_(self.poly_fn) # Only works on CPU
            self.test_set = TensorDataset(feature_samples, target_samples)
        
        if stage == "predict":
            feature_samples = self.data_dist.sample((self.n_data,1))
            target_samples = feature_samples.clone().apply_(self.poly_fn) # Only works on CPU
            self.pred_set = TensorDataset(feature_samples, target_samples)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=32, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=32, num_workers=4)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=32, num_workers=4)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_set, batch_size=32, num_workers=4)

if __name__ == "__main__":
    poly = lambda x: x**2
    data = PolynomialGenerator(poly, 50000)
    data.setup("fit")
    train = data.train_dataloader()
