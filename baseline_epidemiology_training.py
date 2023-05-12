from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from utils.logging_helper import baseline_epidemiology_metrics
from data.epidemiology import *
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

parser = ArgumentParser()

# Data arguments
parser.add_argument("--seq_len", type=int, default=2)
parser.add_argument("--lag_size", type=int, default=1)
parser.add_argument("--poly_features", action="store_true")
parser.add_argument("--apply_scaling", action="store_true")
parser.add_argument("--n_data", type=int, default=100000)
parser.add_argument("--n_degree", type=int, default=3)

# Model arguments
parser.add_argument("--model", type=str, required=True)

# Parse arguments
args = parser.parse_args()
assert args.seq_len >= 2, "Sequence must be two or longer."

# Initialize dataloader and create relevant logging
dataloader = EpidemiologyModule(n_data=args.n_data,
                                batch_size=args.n_data, 
                                lag_size=args.lag_size, 
                                seq_len=args.seq_len,
                                poly_features=args.poly_features, 
                                n_degree_poly=args.n_degree)
log_name = type(dataloader).__name__ + "/"

# Choose model
try:
    if args.model == "GradientBoostingRegressor": # Does not support multiple outputs natively
        model = MultiOutputRegressor(eval(args.model, globals())())
    else:
        model = eval(args.model, globals())()
except:
    raise NotImplementedError("The model {n} is not implemented or imported correctly.")

log_name += "Baselines/" + type(model).__name__

# Add polynomial features for data if relevant
if args.poly_features:
    log_name += "_polynomial"

# Initialize logger
logger = TensorBoardLogger("tb_logs", name=log_name)
run_kwargs = {"polynomial_features": args.poly_features,
              "n_degree": args.n_degree}

# Training
dataloader.setup("fit")
train_dataloader = dataloader.train_dataloader()
beta, gamma, X_train, y_train = next(iter(train_dataloader))
beta = beta.unsqueeze(-1).repeat(1,X_train.size(1)).unsqueeze(-1).to(torch.float32)
gamma = gamma.unsqueeze(-1).repeat(1,X_train.size(1)).unsqueeze(-1).to(torch.float32)
X_train = torch.concat([X_train, beta, gamma], dim=-1)
X_train, y_train = X_train.flatten(0,1), y_train.flatten(0,1)
model.fit(X_train, y_train)

# Validation
val_dataloader = dataloader.val_dataloader()
beta, gamma, X_val, y_val = next(iter(val_dataloader))
beta = beta.reshape(len(X_val), 1).to(torch.float32)
gamma = gamma.reshape(len(X_val), 1).to(torch.float32)
X_forward = torch.concat([X_val[:, 0, :], beta, gamma], dim=1)
y_pred = torch.empty_like(y_val)
for i in range(X_val.shape[1]):
    next_state = torch.Tensor(model.predict(X_forward))
    y_pred[:, i, :] = next_state
    X_forward = torch.concat([next_state, beta, gamma], dim=1)
 
y_val = y_val.flatten(0,1)
y_pred = y_pred.flatten(0,1)
print(args.model)
baseline_epidemiology_metrics(logger, y_val, y_pred, "fit")

# Testing
dataloader.setup("test")
test_dataloader = dataloader.test_dataloader()
beta, gamma, X_test, y_test = next(iter(test_dataloader))
beta = beta.reshape(len(X_test), 1).to(torch.float32)
gamma = gamma.reshape(len(X_test), 1).to(torch.float32)
X_forward = torch.concat([X_test[:, 0, :], beta, gamma], dim=1)
y_pred = torch.empty_like(y_test)
for i in range(X_test.shape[1]):
    next_state = torch.Tensor(model.predict(X_forward))
    y_pred[:, i, :] = next_state
    X_forward = torch.concat([next_state, beta, gamma], dim=1)

y_test = y_test.flatten(0,1)
y_pred = y_pred.flatten(0,1)
baseline_epidemiology_metrics(logger, y_test, y_pred, "test")