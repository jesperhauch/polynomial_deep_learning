import torch.distributions as dist
from typing import Tuple
import torch
import numpy as np

class SteelColumn:
    """Steel Column Function from https://www.sfu.ca/~ssurjano/steelcol.html
    """
    train_quantiles = torch.tensor([0.05, 0.95]) # assumes 90/10 train-test split
    # Fixed values
    L = 7500 # mm
    # Eldred et al. (2008) use values
    B, D, H = 300, 20, 300
    def __init__(self, n_data: int, noise: bool = False):
        self.noise = noise
        self.train_test_split(n_data)

        F_s, _ = dist.log_normal.LogNormal(np.log(400), np.log(35)).sample((n_data,)).sort() # yield stress (MPa)
        P_1, _ = dist.normal.Normal(500000, 50000).sample((n_data,)).sort() # dead weight load (N)
        P_2, _ = dist.gumbel.Gumbel(600000, 90000).sample((n_data,)).sort() # variable load (N)
        P_3, _ = dist.gumbel.Gumbel(600000, 90000).sample((n_data,)).sort() # variable load (N)
        F_0, _ = dist.normal.Normal(30, 10).sample((n_data,)).sort() # initial deflection (mm)
        E, _ = dist.weibull.Weibull(210000., 4200.).sample((n_data,)).sort() # Young's modulus (MPa)
        P = P_1 + P_2 + P_3
        E_b = (torch.pi**2*E*self.B*self.D*self.H**2)/(2*self.L**2)
        self.feature_samples = torch.stack([F_s, F_0, E_b, P], dim=1)   

    def train_test_split(self, n_data):
        self.train_indices = torch.arange(*(self.train_quantiles*n_data)).to(torch.int64)
        self.test_indices = torch.cat([torch.arange(0, self.train_indices[0]), torch.arange(self.train_indices[-1]+1, n_data)]).to(torch.int64)

    def calculate_targets(self, F_s, F_0, E_b, P):
        term_1 = 1 / (2*self.B*self.D)
        term_2 = F_0*E_b / (self.B*self.D*self.H*(E_b-P))
        target_samples = F_s - P*(term_1 + term_2)
        if self.noise:
            target_samples = target_samples + dist.normal.Normal(0,1).sample((len(target_samples),))
        return target_samples

    def generate_train_data(self) -> Tuple:
        shuffle_idx = torch.randperm(len(self.train_indices))
        train_set = self.feature_samples[self.train_indices,:][shuffle_idx,:]
        return train_set, self.calculate_targets(*train_set.permute(1,0))
    
    def generate_test_data(self) -> Tuple:
        shuffle_idx = torch.randperm(len(self.test_indices))
        train_set = self.feature_samples[self.test_indices,:][shuffle_idx,:]
        return train_set, self.calculate_targets(*train_set.permute(1,0))

class SulfurModel(SteelColumn):
    """Sulfur Model Function from https://www.sfu.ca/~ssurjano/sulf.html
    """
    # Fixed values
    S_0 = 1366 # W/m2
    A = 5.1*10**14 # m2
    def __init__(self, n_data: int, noise: bool = False):
        SteelColumn.__init__(self, n_data, noise)
        #Uncertainties
        T_r = dist.log_normal.LogNormal(0.76, 1.2).sample((n_data,)) # transmittance of the atmospheric layer above the aerosol layer
        A_c = dist.log_normal.LogNormal(0.39, 1.1).sample((n_data,)) # fractional cloud cover
        R_s = dist.log_normal.LogNormal(0.85, 1.1).sample((n_data,)) # mean albedo of the underlying surface
        beta = dist.log_normal.LogNormal(0.3, 1.3).sample((n_data,)) # backscattered fraction
        Psi_e = dist.log_normal.LogNormal(5.0, 1.4).sample((n_data,)) # mass scattering efficiency (m^2g^(-1))
        f_Psi_e = dist.log_normal.LogNormal(1.7, 1.2).sample((n_data,)) # scaling factor for Psi_e
        Q = dist.log_normal.LogNormal(71.0, 1.15).sample((n_data,)) # global input flux of anthropogenic sulfur (10^12 g*y*r^(-1))
        Y = dist.log_normal.LogNormal(0.5, 1.5).sample((n_data,)) # fraction of sulfur dioxide oxidized to sulfate aerosol
        L = dist.log_normal.LogNormal(5.5, 1.5).sample((n_data,))
        self.feature_samples = torch.stack([T_r, A_c, R_s, beta, Psi_e, f_Psi_e, Q, Y, L], dim=1)

    def calculate_targets(self, T_r, A_c, R_s, beta, Psi_e, f_Psi_e, Q, Y, L):
        term1 = 3*Q*Y*L/self.A
        target_samples = -0.5*self.S_0**2*(1-A_c)*T_r**2*(1-R_s)**2*beta*Psi_e*f_Psi_e*term1
        if self.noise:
            target_samples = target_samples + dist.normal.Normal(0,1).sample((len(target_samples),))
        return target_samples

class ShortColumn(SteelColumn):
    # Fixed values
    b, h = 5, 15
    Y = dist.log_normal.LogNormal(5, 0.5).sample() # yield stress

    def __init__(self, n_data: int, noise: bool = False):
        SteelColumn.__init__(self, n_data, noise)

        # Uncertainties
        M = dist.normal.Normal(2000, 400).sample((n_data,)) # bending moment
        P = dist.normal.Normal(500, 100).sample((n_data,)) # axial force

        self.feature_samples = torch.stack([M, P], dim=1)

    def calculate_targets(self, M, P):
        term1 = 4*M/(self.b*self.h**2*self.Y)
        term2 = P**2/(self.b**2*self.h**2*self.Y**2)
        target_samples = 1-term1-term2
        if self.noise:
            target_samples = target_samples + dist.normal.Normal(0,1).sample((len(target_samples),))
        return target_samples

if __name__ == "__main__":
    data = SteelColumn(100000)
    print(data.generate_train_data()[0].shape)
    print(data.generate_test_data()[0].shape)
    data = SulfurModel(100000)
    print(data.generate_train_data()[0].shape)
    print(data.generate_test_data()[0].shape)
    data = ShortColumn(100000)
    print(data.generate_train_data()[0].shape)
    print(data.generate_test_data()[0].shape)


