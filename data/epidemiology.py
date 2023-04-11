from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import pytorch_lightning as pl
from typing import List, Tuple

class Epidemiology(Dataset):
    """https://allendowney.github.io/ModSimPy/chap11.html

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, beta: float, gamma: float, seq_len: int, n_data: int, vac_times: List = [], vac_fraction: float = 0.0):
        self.beta = beta
        self.gamma = gamma
        self.vac_times = vac_times
        self.vac_fraction = vac_fraction
        self.sequences = [self.generate_sequence(seq_len) for _ in range(n_data)]

    def __len__(self):
        return len(self.sequences)

    def generate_sequence(self, seq_len: int):
        random_s = torch.randint(low=50, high=100, size=(1,)).item()
        random_i = torch.randint(low=0, high=10, size=(1,)).item()
        s = random_s / (random_i+random_s)
        i = random_i / (random_i+random_s)
        r = 0
        #self.s = [s]
        self.i = [i]
        #self.r = [r]
        for t in range(1, seq_len):
            if t in self.vac_times:
                s, r = self.add_immunization(self.vac_fraction)
            infected = self.beta * i * s
            recovered = self.gamma * i
            s -= infected
            i += infected - recovered
            r += recovered

            #self.s.append(s)
            self.i.append(i)
            #self.r.append(r)
        return torch.Tensor(self.i)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index][:-1], self.sequences[index][1:]

    def add_immunization(self, s: float, r: float, fraction: float):
        return s - fraction, r + fraction

class SimulationModule(pl.LightningDataModule): # TODO: Fix class to generate 
    train_ratio = 0.8
    test_ratio = 0.1
    def __init__(self, batch_size: int = 32, **kwargs) -> None:
        super().__init__()
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = False
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = Epidemiology(1/80, 1/160, 500, 10000)
            self.val_set = Epidemiology(1/80, 1/160, 100, 1000)
        if stage == "test":
            self.test_set = Epidemiology(1/40, 1/90, 100, 1000)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)


if __name__ == "__main__":
    ds = Epidemiology(1/80, 1/160, 500, 5)
    #ds2 = EpidemiologyDataset(1/80, 1/160, 1000, [], None)
    data_module = SimulationModule(64)
    data_module.setup("fit")
    train = data_module.train_dataloader()
    x,y = next(iter(train))
    pass
