class EpidemiologyModelWrapper(pl.LightningModule): # Incorrect model
    loss = nn.MSELoss()
    def __init__(self, temporal_layer: nn.Module, multiplication_net: nn.Module, input_size: int, hidden_size: int, n_degree: int, n_layers: int = 1, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        setattr(self, "s_net", multiplication_net(hidden_size, hidden_size, n_degree, 1))
        setattr(self, "i_net", multiplication_net(hidden_size, hidden_size, n_degree, 1))
        setattr(self, "temporal_layer", temporal_layer(input_size, hidden_size, n_layers, batch_first=True))
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def init_hidden(self, batch_size: int):
        n_hidden = 1 if type(self.temporal_layer).__name__ in ["RNN", "GRU"] else 2 # only works for RNN, GRU and LSTM
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size) for _ in range(n_hidden))
        #return (nn.init.kaiming_uniform_(torch.empty(self.n_layers, batch_size, self.hidden_size)) for _ in range(n_hidden)) # He initialization
    
    def forward(self, x: torch.Tensor, *args) -> Tuple[torch.Tensor]:
        x, hidden = self.temporal_layer(x, *args)
        s = self.s_net(x)
        i = self.i_net(x)
        return torch.sigmoid(torch.stack((s,i), dim=-1)), hidden
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        hidden = self.init_hidden(X.size(0))
        y_hat, (hidden) = self(X, *hidden)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        hidden = self.init_hidden(X.size(0))
        y_hat = torch.zeros_like(y)
        X_forward = X[:, 0, :]
        for t in range(X.size(1)):
            X_forward, hidden = self(X_forward, *hidden)
            y_hat[:, t, :] = X_forward
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)

        # Extra logging metrics - need to flatten batch dimension
        y = y.flatten(0,1)
        y_hat = y_hat.flatten(0,1)
        self.log('val_r2', r2_score(y, y_hat), on_epoch=True)
        self.log('val_rmse', mean_squared_error(y, y_hat, squared=False), on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        hidden = self.init_hidden(X.size(0))
        y_hat = torch.zeros_like(y)
        X_forward = X[:, 0, :]
        for t in range(X.size(1)):
            X_forward, hidden = self(X_forward, *hidden)
            y_hat[:, t, :] = X_forward
        # Extra logging metrics - need to flatten batch dimension
        y = y.flatten(0,1)
        y_hat = y_hat.flatten(0,1)
        self.log('test_r2', r2_score(y, y_hat))
        self.log('test_mae', mean_absolute_error(y, y_hat))
        self.log("test_mape", mean_absolute_percentage_error(y, y_hat))
        self.log("test_rmse", mean_squared_error(y, y_hat, squared=False))