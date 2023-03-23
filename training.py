from models.neural_nets import *
from models.pi_nets import *
from data.polynomials import PolynomialGenerator
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

parser = ArgumentParser()

# Data arguments
parser.add_argument("--polynomial", type=str, required=True)
parser.add_argument("--polynomial_name", type=str, required=True)
parser.add_argument("--data_mean", type=int, default=0)
parser.add_argument("--data_std", type=float, default=5.0)
parser.add_argument("--noise", action="store_true")
parser.add_argument("--n_data", type=int, default=100000)

# Model arguments
parser.add_argument("--model", type=str, choices=["ccp", "polynomial_nn"], required=True)
parser.add_argument("--n_neurons", type=int, default=64)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_degree", type=int, default=2)
parser.add_argument("--relu", action="store_true")

# Trainer/logger arguments
parser.add_argument("--epochs", type=int, default=30)

# Parse arguments
args = parser.parse_args()
polynomial = eval(args.polynomial, globals())
in_dim = polynomial.__code__.co_argcount

# Initialize dataloader
if args.noise:
    noise = torch.distributions.normal.Normal(0,1)
else:
    noise = None
dataloader = PolynomialGenerator(polynomial, args.n_data, in_dim, torch.distributions.normal.Normal(args.data_mean, args.data_std), noise)

# Choose model
if args.model == "polynomial_nn":
    if args.relu:
        model = PolynomialNN_relu(in_dim, [args.n_neurons]*args.n_layers, 1, args.n_degree)
    else:    
        model = PolynomialNN(in_dim, [args.n_neurons]*args.n_layers, 1, args.n_degree)
        #model = PolynomialNN_each(in_dim, [args.n_neurons]*args.n_layers, 1)
elif args.model == "ccp":
    if args.relu:
        model = CCP_relu(in_dim, args.n_neurons, args.n_degree, 1)
    else:    
        model = CCP(in_dim, args.n_neurons, args.n_degree, 1)

# Initialize logger and Trainer
noise_name = "_noise" if args.noise else ""
log_name = f"{args.polynomial_name}{noise_name}/{type(model).__name__}"
    
logger = pl.loggers.TensorBoardLogger("tb_logs", name=log_name)
trainer = pl.Trainer(limit_train_batches=64,max_epochs=args.epochs, log_every_n_steps=25, logger=logger)

# Log data hyperparams
logger.log_hyperparams({"data_dist": str(dataloader.data_dist), "noise": str(dataloader.noise)})

# Start training
trainer.fit(model=model, datamodule=dataloader)
trainer.logger.finalize("success")
trainer.logger.save()
print("=====================================================")
print("Polynomial:", args.polynomial_name)
print("Model:", model)
print("Train distribution: Normal({mean}, {std})".format(mean=args.data_mean, std=args.data_std))
print("Non-deterministic:", args.noise)
print("=====================================================")
trainer.test(model=model, datamodule=dataloader)