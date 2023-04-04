from models.neural_nets import *
from models.pi_nets import *
from utils.model_utils import *
from utils.logging_helper import run_information
from data.polynomials import *
from data.simulation_functions import *
import pytorch_lightning as pl
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
parser.add_argument("--n_neurons", type=int, default=64)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_degree", type=int, default=2)
parser.add_argument("--out_dim", type=int, default=1)
parser.add_argument("--relu", action="store_true")

# Trainer/logger arguments
parser.add_argument("--epochs", type=int, default=30)

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
    log_name = f"Simulations/{type(data_gen).__name__}"

dataloader = PolynomialModule(data_gen)
log_name += "_noise/" if args.noise else "/"

# Choose model
model_args = {"input_size": in_dim,
              "hidden_sizes": [args.n_neurons]*args.n_layers,
              "n_degree": args.n_degree,
              "output_size": args.out_dim}
try:
    model = eval(args.model, globals())(**model_args)
except:
    raise NotImplementedError("The model {n} is not implemented or imported correctly.")

# Initialize logger and Trainer
log_name += type(model).__name__
logger = pl.loggers.TensorBoardLogger("tb_logs", name=log_name)
trainer = pl.Trainer(limit_train_batches=64,max_epochs=args.epochs, log_every_n_steps=25, logger=logger)

# Log data hyperparams and output run information
run_information(logger, data_gen, model, **data_args)

# Start training
trainer.fit(model=model, datamodule=dataloader)
trainer.logger.finalize("success")
trainer.logger.save()
trainer.test(model=model, datamodule=dataloader)