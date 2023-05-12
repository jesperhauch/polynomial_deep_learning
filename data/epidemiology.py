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
    def __init__(self, beta_distribution: Distribution, gamma_distribution: Distribution, n_data: int, seq_len: int, lag_size: int, poly_features: bool = False, n_degree_poly: int = 2, **kwargs):
        self.poly_features = poly_features
        self.n_degree_poly = n_degree_poly
        self.sequences = [self.generate_sequence(beta_distribution, gamma_distribution, seq_len, lag_size, i) for i in range(n_data)]

    def __len__(self):
        return len(self.sequences)
    
    def simulation_length_assertion(self):
        assert len(self.s) > 1, "Please choose sequence length and lag size that ensures a simulation with more than one timestep."
        assert len(self.i) > 1, "Please choose sequence length and lag size that ensures a simulation with more than one timestep."
        assert len(self.r) > 1, "Please choose sequence length and lag size that ensures a simulation with more than one timestep."

    def start_state(self):
        state_sum = 0
        while state_sum == 0: # ensure no states with all zeros
            random_s = torch.randint(low=0, high=100, size=(1,)).item()
            random_i = torch.randint(low=0, high=100, size=(1,)).item()
            random_r = torch.randint(low=0, high=100, size=(1,)).item()
            state_sum = random_s + random_i + random_r

        s = random_s / (random_s+random_i+random_r)
        i = random_i / (random_s+random_i+random_r)
        r = random_r / (random_s+random_i+random_r)
        return s, i, r

    def generate_sequence(self, beta_distribution: Distribution, gamma_distribution: Distribution, seq_len: int, lag_size: int, seed: int):
        torch.manual_seed(seed)
        beta = beta_distribution.sample().item()
        gamma = gamma_distribution.sample().item()
        s, i, r = self.start_state()
        self.s = [s]
        self.i = [i]
        self.r = [r]
        for t in range(1, seq_len):
            infected = beta * i * s
            recovered = gamma * i
            s -= infected
            i += infected - recovered
            r += recovered

            if t % lag_size == 0:
                self.s.append(s)
                self.i.append(i)
                self.r.append(r)
        self.simulation_length_assertion()
        return {"sequence": torch.stack([torch.Tensor(self.s), torch.Tensor(self.i), torch.tensor(self.r)], dim=1),
                "beta": beta,
                "gamma": gamma}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target = self.sequences[index]['sequence'][1:]
        features = self.sequences[index]['sequence'][:-1]
        beta, gamma = self.sequences[index]['beta'], self.sequences[index]['gamma']
        if self.poly_features:
            poly = PolynomialFeatures(degree=self.n_degree_poly, include_bias=False)
            features = features.numpy()
            features = poly.fit_transform(features)
            features = torch.Tensor(features)
        return beta, gamma, features, target

class EpidemiologyModule(LightningDataModule):
    train_ratio = 0.8
    test_ratio = 0.1
    beta_train_distribution = torch.distributions.uniform.Uniform(0.02, 0.09)
    gamma_train_distribution = torch.distributions.uniform.Uniform(0.002, 0.01)
    beta_test_distribution = torch.distributions.uniform.Uniform(0.1, 0.25)
    gamma_test_distribution = torch.distributions.uniform.Uniform(0.05, 0.1)
    def __init__(self, n_data: int = 100000, batch_size: int = 32, seq_len: int = 1, lag_size: int = 1, poly_features: bool = False, n_degree_poly: int = 2, **kwargs) -> None:
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = False
        self.n_data = n_data
        self.batch_size = batch_size
        self.lag_size = lag_size
        self.seq_len = seq_len
        self.poly_features = poly_features
        self.n_degree_poly = n_degree_poly
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = Epidemiology(self.beta_train_distribution, self.gamma_train_distribution, int(self.train_ratio*self.n_data), self.seq_len, self.lag_size, self.poly_features, self.n_degree_poly)
            self.val_set = Epidemiology(self.beta_train_distribution, self.gamma_train_distribution, int(self.test_ratio*self.n_data), self.seq_len, self.lag_size, self.poly_features, self.n_degree_poly)
        if stage == "test":
            self.test_set = Epidemiology(self.beta_test_distribution, self.gamma_test_distribution, int(self.test_ratio*self.n_data), self.seq_len, self.lag_size, self.poly_features, self.n_degree_poly)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

def generate_sir_from_state(s_t: float, i_t: float, r_t: float, beta: float, gamma: float, seq_len: int, lag_size: int):
        s = [s_t]
        i = [i_t]
        r = [r_t]
        for t in range(1, seq_len-1):
            infected = beta * i_t * s_t
            recovered = gamma * i_t
            s_t -= infected
            i_t += infected - recovered
            r_t += recovered

            if t % lag_size == 0:
                s.append(s_t)
                i.append(i_t)
                r.append(r_t)
        return torch.stack([torch.Tensor(s), torch.Tensor(i), torch.tensor(r)], dim=1)


if __name__ == "__main__":
    dataloader = EpidemiologyModule(100, 32, 2, 1)
    dataloader.setup("fit")
    train = dataloader.train_dataloader()
    for X, y in train:
        break
    #X1, y1 = data.__getitem__(0)
    pass