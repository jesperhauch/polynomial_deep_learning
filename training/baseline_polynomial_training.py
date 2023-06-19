from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from utils.logging_helper import baseline_run_information, baseline_metrics
from data.polynomials import *
from data.optimization_functions import *

from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

parser = ArgumentParser()

# Data arguments
parser.add_argument("--polynomial", type=str)
parser.add_argument("--polynomial_name", type=str)
parser.add_argument("--data_generator", type=str, default="Normal")
parser.add_argument("--data_mean", type=int, default=0)
parser.add_argument("--data_std", type=float, default=5.0)
parser.add_argument("--noise", action="store_true")
parser.add_argument("--standardize", action="store_true")
parser.add_argument("--n_data", type=int, default=100000)

# Model arguments
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--n_degree", type=int, default=2)
parser.add_argument("--polynomial_features", action="store_true")

# Parse arguments
args = parser.parse_args()

# Initialize dataloader and create relevant logging
if args.data_generator == "Normal":
    polynomial = eval(args.polynomial, globals())
    in_dim = polynomial.__code__.co_argcount
    data_gen = NormalGenerator(polynomial, in_dim, args.n_data, args.data_mean, args.data_std, args.noise)
    data_args = {"data_dist": str(data_gen.data_dist),
                 "noise": str(args.noise),
                 "polynomial_name": args.polynomial_name}
    log_name = args.polynomial_name
else: 
    try:
        data_gen = eval(args.data_generator, globals())(args.n_data, args.noise, args.standardize)
        in_dim = data_gen.n_features()
    except:
        raise NotImplementedError("The generator you are looking for is not implemented.")
    data_args = {"noise": args.noise}
    log_name = f"Optimization/{type(data_gen).__name__}"

dataloader = PolynomialModule(fn_data=data_gen, batch_size=args.n_data)
log_name += "_noise/" if args.noise else "/"

# Choose model
try:
    model = eval(args.model, globals())()
except:
    raise NotImplementedError("The model {n} is not implemented or imported correctly.")

log_name += type(model).__name__

# Add polynomial features for data if relevant
if args.polynomial_features:
    model = Pipeline([("poly", PolynomialFeatures(degree=args.n_degree)),
                      ("model_component", model)])
    log_name += "_polynomial"

# Initialize logger
logger = TensorBoardLogger("tb_logs", name=log_name)
run_kwargs = {"polynomial_features": args.polynomial_features,
              "n_degree": args.n_degree}
baseline_run_information(data_gen, model, **run_kwargs)

# Training and validation
dataloader.setup("fit")
train_dataloader = dataloader.train_dataloader()
X_train, y_train = next(iter(train_dataloader))
model.fit(X_train, y_train)
val_dataloader = dataloader.val_dataloader()
X_val, y_val = next(iter(val_dataloader))
y_pred = model.predict(X_val)
baseline_metrics(logger, y_val, y_pred, "fit")

# Testing
dataloader.setup("test")
test_dataloader = dataloader.test_dataloader()
X_test, y_test = next(iter(*test_dataloader))
y_pred = model.predict(X_test)
baseline_metrics(logger, y_test, y_pred, "test")
print("Done")