import torch.distributions as dist
from data.base_data import BaseDataClass
import torch
import numpy as np

class SteelColumn(BaseDataClass):
    """Steel Column Function from https://www.sfu.ca/~ssurjano/steelcol.html
    """
    # Fixed values
    L = 7500 # mm
    # Eldred et al. (2008) use values
    B, D, H = 300, 20, 300
    def build_features(self, n_data: int) -> torch.Tensor:
        sample_shape = torch.empty(n_data,)
        F_s = dist.log_normal.LogNormal(np.log(400), np.log(35)).sample(sample_shape.size()) # yield stress (MPa)
        P_1 = dist.normal.Normal(500000, 50000).sample(sample_shape.size()) # dead weight load (N)
        P_2 = dist.gumbel.Gumbel(600000, 90000).sample(sample_shape.size()) # variable load (N)
        P_3 = dist.gumbel.Gumbel(600000, 90000).sample(sample_shape.size()) # variable load (N)
        F_0 = dist.normal.Normal(30, 10).sample(sample_shape.size()) # initial deflection (mm)
        E = dist.weibull.Weibull(210000., 4200.).sample(sample_shape.size()) # Young's modulus (MPa)
        P = P_1 + P_2 + P_3
        pi = torch.Tensor(torch.pi)
        E_b = (torch.pow(pi, 2)*E*self.B*self.D*self.H**2)/(2*self.L**2)
        return torch.stack((F_s, F_0, E_b, P), dim=1)

    def calculate_targets(self, F_s, F_0, E_b, P):
        term_1 = 1 / (2*self.B*self.D)
        term_2 = F_0*E_b / (self.B*self.D*self.H*(E_b-P))
        return F_s - P*(term_1 + term_2)

class SulfurModel(BaseDataClass):
    """Sulfur Model Function from https://www.sfu.ca/~ssurjano/sulf.html
    """
    # Fixed values
    S_0 = 1366 # W/m2
    A = 5.1*10**14 # m2
    def build_features(self, n_data: int) -> torch.Tensor:
        sample_shape = torch.empty(n_data,)
        T_r = dist.log_normal.LogNormal(np.log(0.76), np.log(1.2)).sample(sample_shape.size()) # transmittance of the atmospheric layer above the aerosol layer
        A_c = dist.log_normal.LogNormal(np.log(0.39), np.log(1.1)).sample(sample_shape.size()) # fractional cloud cover
        R_s = dist.log_normal.LogNormal(np.log(0.85), np.log(1.1)).sample(sample_shape.size()) # mean albedo of the underlying surface
        beta = dist.log_normal.LogNormal(np.log(0.3), np.log(1.3)).sample(sample_shape.size()) # backscattered fraction
        Psi_e = dist.log_normal.LogNormal(np.log(5.0), np.log(1.4)).sample(sample_shape.size()) # mass scattering efficiency (m^2g^(-1))
        f_Psi_e = dist.log_normal.LogNormal(np.log(1.7), (1.2)).sample(sample_shape.size()) # scaling factor for Psi_e
        Q = dist.log_normal.LogNormal(np.log(71.0), np.log(1.15)).sample(sample_shape.size()) # global input flux of anthropogenic sulfur (10^12 g*y*r^(-1))
        Y = dist.log_normal.LogNormal(np.log(0.5), np.log(1.5)).sample(sample_shape.size()) # fraction of sulfur dioxide oxidized to sulfate aerosol
        L = dist.log_normal.LogNormal(np.log(5.5), np.log(1.5)).sample(sample_shape.size())
        return torch.stack((T_r, A_c, R_s, beta, Psi_e, f_Psi_e, Q, Y, L), dim=1)

    def calculate_targets(self, T_r, A_c, R_s, beta, Psi_e, f_Psi_e, Q, Y, L) -> torch.Tensor:
        term1 = 3*Q*Y*L/self.A
        expr = -0.5*self.S_0**2*(1-A_c)*T_r**2*(1-R_s)**2*beta*Psi_e*f_Psi_e*term1
        return expr

class ShortColumn(BaseDataClass):
    """Short Column Function from https://www.sfu.ca/~ssurjano/shortcol.html
    """
    # Fixed values
    b, h = 5, 15
    Y = dist.log_normal.LogNormal(5, 0.5).sample() # yield stress
    def build_features(self, n_data: int) -> torch.Tensor:
        sample_shape = torch.empty(n_data,)
        M = dist.normal.Normal(2000, 400).sample(sample_shape.size()) # bending moment
        P = dist.normal.Normal(500, 100).sample(sample_shape.size()) # axial force
        return torch.stack([M, P], dim=1)

    def calculate_targets(self, M, P) -> torch.Tensor:
        term1 = 4*M/(self.b*self.h**2*self.Y)
        term2 = P**2/(self.b**2*self.h**2*self.Y**2)
        return 1-term1-term2

class Bukin06(BaseDataClass):
    """https://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Bukin06

    Args:
        BaseDataClass (_type_): _description_
    """
    def build_features(self, n_data: int) -> torch.Tensor:
        sample_shape = torch.empty(n_data,)
        x_1 = dist.uniform.Uniform(-15,-5).sample(sample_shape.size())
        x_2 = dist.uniform.Uniform(-3,3).sample(sample_shape.size())
        return torch.stack([x_1, x_2], dim=1)

    def calculate_targets(self, x1, x2):
        return 100*torch.sqrt(torch.abs(x2-0.01*x1**2))+0.01*torch.abs(x1+10)

class Currin(BaseDataClass):
    """From https://www.sfu.ca/~ssurjano/curretal91.html

    Args:
        BaseDataClass (_type_): _description_
    """
    def build_features(self, n_data: int) -> torch.Tensor:
        sample_shape = torch.empty(n_data,)
        return dist.uniform.Uniform(0, 1).sample(sample_shape.size())
    
    def calculate_targets(self, x_1, x_2) -> torch.Tensor:
        return 4.9 + 21.15*x_1 - 2.17*x_2 - 15.88*x_1**2 - 1.38*x_2**2 - 5.26*x_1*x_2

class LimPolynomial(Currin):
    """https://www.sfu.ca/~ssurjano/limetal02pol.html

    Args:
        Currin (_type_): _description_
    """
    def calculate_targets(self, x_1, x_2) -> torch.Tensor:
        return 9. + 5/2*x_1 - 35/2*x_2 + 5/2*x_1*x_2 + 19*x_2**2 - 15/2*x_1**3 - 5/2*x_1*x_2 - 11/2*x_2**4 + x_1**3*x_2**2

class Colville(BaseDataClass):
    """https://arxiv.org/pdf/1308.4008.pdf

    Args:
        BaseDataClass (_type_): _description_
    """
    def build_features(self, n_data: int) -> torch.Tensor:
        sample_shape = torch.empty(n_data,4)
        return dist.uniform.Uniform(-10, 10).sample(sample_shape.size())

    def calculate_targets(self, x1, x2, x3, x4) -> torch.Tensor:
        expr = 100*(x1**2-x2)**2 + (x1-1)**2 + (x3-1)**2 + 90*(x3**2-x4)**2 + 10.1*((x2-1)**2 + (x4-1)**2)+19.8*(x2-1)*(x4-1)
        return expr

class DettePepelyshev(BaseDataClass):
    """https://www.sfu.ca/~ssurjano/detpep10curv.html

    Args:
        BaseDataClass (_type_): _description_
    """
    def build_features(self, n_data: int) -> torch.Tensor:
        sample_shape = torch.empty(n_data,3)
        return dist.uniform.Uniform(0, 1).sample(sample_shape.size())
    
    def calculate_targets(self, x1, x2, x3) -> torch.Tensor:
        expr = 4*(x1-2+8*x2-8*x2**2)**2 + (3-4*x2)**2 + 16*torch.sqrt(x3+1)*(2*x3-1)**2
        return expr

class Beale(BaseDataClass):
    """https://arxiv.org/pdf/1308.4008.pdf

    Args:
        BaseDataClass (_type_): _description_
    """
    def build_features(self, n_data: int) -> torch.Tensor:
        sample_shape = torch.empty(n_data,2)
        return dist.uniform.Uniform(-4.5, 4.5).sample(sample_shape.size())

    def calculate_targets(self, x1, x2):
        expr = (x1*x2-x1+1.5)**2 + (x1*x2**2-x1 + 2.25)**2 + (x1*x2**3-x1+2.625)**2
        return expr

class Price03(BaseDataClass):
    """https://arxiv.org/pdf/1308.4008.pdf

    Args:
        BaseDataClass (_type_): _description_
    """
    def build_features(self, n_data: int) -> torch.Tensor:
        sample_shape = torch.empty(n_data,2)
        return dist.uniform.Uniform(-500, 500).sample(sample_shape.size())
    
    def calculate_targets(self, x1, x2):
        return 100*(x2-x1**2)**2+6*(6.4*(x2-0.5)**2-x1-0.6)**2

class CamelThreeHump(BaseDataClass):
    """https://ieeexplore.ieee.org/document/5391457

    Args:
        BaseDataClass (_type_): _description_
    """
    def build_features(self, n_data: int) -> torch.Tensor:
        sample_shape = torch.empty(n_data,2)
        return dist.uniform.Uniform(-5, 5).sample(sample_shape.size())
    
    def calculate_targets(self, x1, x2) -> torch.Tensor:
        return 2*x1**2 - 1.05*x1**4 + (x1**6)/6 + x1*x2 + x2**2
    
class GoldsteinPrice(BaseDataClass):
    """https://arxiv.org/pdf/1308.4008.pdf

    Args:
        BaseDataClass (_type_): _description_
    """
    def build_features(self, n_data: int) -> torch.Tensor:
        sample_shape = torch.empty(n_data,2)
        return dist.uniform.Uniform(-2, 2).sample(sample_shape.size())
    
    def calculate_targets(self, x1, x2) -> torch.Tensor:
        expr = (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))\
                * (30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
        return expr

if __name__ == "__main__":
    data = GoldsteinPrice(100000)
    features, targets = data.generate_train_data()
    print(targets.quantile(torch.Tensor([0, 0.25, 0.5, 0.75, 1])))


