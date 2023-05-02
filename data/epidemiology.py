from torch.utils.data import Dataset, DataLoader
import torch
import pytorch_lightning as pl
from lightning import LightningDataModule
from typing import List, Tuple
from sklearn.preprocessing import PolynomialFeatures
from torch.distributions import Distribution

class Epidemiology(Dataset):
    """Models the number of infected at a timestep from the amount of suscetiple and infected in previous timestep
    https://allendowney.github.io/ModSimPy/chap11.html

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, beta_distribution: Distribution, gamma_distribution: Distribution, seq_len: int, n_data: int, lag_size: int, stage: str, poly_features: bool = False, n_degree_poly: int = 2, **kwargs):
        self.beta = beta_distribution.sample().item()
        self.gamma = gamma_distribution.sample().item()
        self.seq_len = seq_len
        self.n_data = n_data
        self.lag_size = lag_size
        self.stage = stage
        self.poly_features = poly_features
        self.n_degree_poly = n_degree_poly

    def __len__(self):
        return self.n_data
    
    def simulation_length_assertion(self):
        assert len(self.s) > 1, "Please choose sequence length and lag size that ensures a simulation with more than one timestep."
        assert len(self.i) > 1, "Please choose sequence length and lag size that ensures a simulation with more than one timestep."
        assert len(self.r) > 1, "Please choose sequence length and lag size that ensures a simulation with more than one timestep."

    def generate_sequence(self, seq_len: int, beta: float, gamma: float, seed: int, stage: str):
        random_s = torch.randint(low=25, high=100, size=(1,)).item()
        random_i = torch.randint(low=1, high=10, size=(1,)).item()
        random_r = 0
        if stage == "train":
            random_r = torch.randint(low=0, high=10, size=(1,)).item()
            
        s = random_s / (random_s+random_i+random_r)
        i = random_i / (random_s+random_i+random_r)
        r = random_r / (random_s+random_i+random_r)
        self.s = [s]
        self.i = [i]
        self.r = [r]
        for t in range(1, seq_len):
            infected = beta * i * s
            recovered = gamma * i
            s -= infected
            i += infected - recovered
            r += recovered

            if t % self.lag_size == 0:
                self.s.append(s)
                self.i.append(i)
                self.r.append(r)
        self.simulation_length_assertion()
        return torch.stack([torch.Tensor(self.s), torch.Tensor(self.i), torch.tensor(self.r)], dim=1)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.generate_sequence(self.seq_len, self.beta, self.gamma, index, self.stage)
        target = sequence[1:]
        features = sequence[:-1]
        if self.poly_features:
            poly = PolynomialFeatures(degree=self.n_degree_poly, include_bias=False)
            features = features.numpy()
            features = poly.fit_transform(features)
            features = torch.Tensor(features)
        return features, target

class EpidemiologyModule(LightningDataModule):
    train_ratio = 0.8
    test_ratio = 0.1
    beta_train_distribution = torch.distributions.uniform.Uniform(0.02, 0.09)
    gamma_train_distribution = torch.distributions.uniform.Uniform(0.002, 0.01)
    beta_test_distribution = torch.distributions.uniform.Uniform(0.1, 0.25)
    gamma_test_distribution = torch.distributions.uniform.Uniform(0.05, 0.1)
    def __init__(self, batch_size: int = 32, lag_size: int = 1, seq_len: int = 1, poly_features: bool = False, n_degree_poly: int = 2, **kwargs) -> None:
        super().__init__()
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = False
        self.batch_size = batch_size
        self.lag_size = lag_size
        self.seq_len = seq_len
        self.poly_features = poly_features
        self.n_degree_poly = n_degree_poly
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = Epidemiology(self.beta_train_distribution, self.gamma_train_distribution, self.seq_len, 80000, self.lag_size, "train", self.poly_features, self.n_degree_poly)
            self.val_set = Epidemiology(self.beta_train_distribution, self.gamma_train_distribution, self.seq_len, 10000, self.lag_size, "val", self.poly_features, self.n_degree_poly)
        if stage == "test":
            self.test_set = Epidemiology(self.beta_test_distribution, self.gamma_test_distribution, self.seq_len, 10000, self.lag_size, "test", self.poly_features, self.n_degree_poly)
        if stage == "predict":
            self.predict_set = Epidemiology(self.beta_test_distribution, self.gamma_test_distribution, self.seq_len, 10000, self.lag_size, "predict", self.poly_features, self.n_degree_poly)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_set, batch_size=self.batch_size, shuffle=False, num_workers=0)


if __name__ == "__main__":
    beta_dist = torch.distributions.uniform.Uniform(0.02, 0.09)
    gamma_dist = torch.distributions.uniform.Uniform(0.002, 0.01)
    #data = Epidemiology(beta_dist, gamma_dist, 2, 100, 1, "train")
    data2 = Epidemiology(beta_dist, gamma_dist, 100, 10, 1, "val")
    #X1, y1 = data.__getitem__(0)
    X2, y2 = data2.__getitem__(0)
    pass