import torch
import torch.nn as nn
from sklearn.preprocessing import PolynomialFeatures

class LinearRegression(nn.Module):
    def __init__(self, input_size: int, output_size: int, **kwargs):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


if __name__ == "__main__":
    from data.epidemiology import EpidemiologyModule
    data_module = EpidemiologyModule(64, 3)
    data_module.setup("fit")
    train = data_module.train_dataloader()
    X,y = next(iter(train))
    model = LinearRegression(2, 1, 3, True)
    #hidden = model.init_hidden(X.size(0))
    #model.forward(X, (hidden))
    model(X)