from torch.utils.data import Dataset
import torch
from typing import List

class EpidemiologyDataset(Dataset):
    def __init__(self, beta: float, gamma: float, t_end: int, vac_times: List = [], vac_fraction: float = 0.0):
        self.beta = beta
        self.gamma = gamma
        self.t_end = t_end
        self.vac_times = vac_times
        self.vac_fraction = vac_fraction

    def __len__(self):
        return self.t_end

    def __getitem__(self, index: int) -> tuple:
        torch.manual_seed(index)
        random_s = torch.randint(low=50, high=100, size=(1,)).item()
        random_i = torch.randint(low=0, high=10, size=(1,)).item()
        self.s = random_s / (random_i+random_s)
        self.i = random_i / (random_i+random_s)
        self.r = 0
        simulation = []
        for t in range(0, self.t_end):
            if t in self.vac_times:
                self.add_immunization(self.vac_fraction)
            infected = self.beta * self.i * self.s
            recovered = self.gamma * self.i
            self.s -= infected
            self.i += infected - recovered
            self.r += recovered
            simulation.append(torch.tensor([self.s, self.i, self.r]))
        output = torch.cat(simulation, dim=0).reshape(self.t_end,3)
        return output[:, [0,2]], output[:, 1]

    def add_immunization(self, fraction: float):
        self.s -= fraction
        self.r += fraction

if __name__ == "__main__":
    ds = EpidemiologyDataset(1/5, 1/9, 25, [], None)
    print(ds.__getitem__(0))
