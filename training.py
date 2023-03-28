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
parser.add_argument("--data_generator", type=str, default="normal")
parser.add_argument("--data_mean", type=int, default=0)
parser.add_argument("--data_std", type=float, default=5.0)
parser.add_argument("--noise", action="store_true")
parser.add_argument("--n_data", type=int, default=100000)

# Model arguments
parser.add_argument("--model", type=str, choices=["ccp", "polynomial_nn", "ffnn", "pdc", "pdclow"], required=True)
parser.add_argument("--n_neurons", type=int, default=64)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_degree", type=int, default=2)
parser.add_argument("--relu", action="store_true")

# Trainer/logger arguments
parser.add_argument("--epochs", type=int, default=30)

# Parse arguments
args = parser.parse_args()

# Initialize dataloader and create relevant logging
if args.data_generator == "normal":
    polynomial = eval(args.polynomial, globals())
    in_dim = polynomial.__code__.co_argcount
    data_gen = NormalGenerator(polynomial, in_dim, args.n_data, args.data_mean, args.data_std, args.noise)
    dataloader = PolynomialModule(data_gen)

    data_args = {"data_dist": str(data_gen.data_dist),
                 "noise": str(args.noise),
                 "polynomial_name": args.polynomial_name}
else: 
    try:
        data_gen = eval(args.data_generator, globals())(args.n_data, args.noise)
    except:
        raise NotImplementedError("The generator you are looking for is not implemented.")
    data_args = {"noise": args.noise}


# Choose model
if args.model == "polynomial_nn":
    if args.relu:
        model = PolynomialNN_relu(in_dim, [args.n_neurons]*args.n_layers, 1, args.n_degree)
    else:    
        model = PolynomialNN(in_dim, [args.n_neurons]*args.n_layers, 1, args.n_degree)
elif args.model == "ccp":
    if args.relu:
        model = CCP_relu(in_dim, args.n_neurons, args.n_degree, 1)
    else:    
        model = CCP(in_dim, args.n_neurons, args.n_degree, 1)
elif args.model == "ffnn":
    model = FeedForwardNN(in_dim, [args.n_neurons]*args.n_layers, 1)
elif args.model == "pdc":
    model = PDC(in_dim, args.n_neurons, args.n_degree, 1)
elif args.model == "pdclow":
    model = PDCLow(in_dim, args.n_neurons, args.n_degree, 1)

# Initialize logger and Trainer TODO: Place this for each data loader
noise_name = "_noise" if args.noise else ""
log_name = f"{args.polynomial_name}{noise_name}/{type(model).__name__}"
    
logger = pl.loggers.TensorBoardLogger("tb_logs", name=log_name)
trainer = pl.Trainer(limit_train_batches=64,max_epochs=args.epochs, log_every_n_steps=25, logger=logger)

# Log data hyperparams and output run information
run_information(logger, data_gen, model, **data_args)

# Start training
trainer.fit(model=model, datamodule=dataloader)
trainer.logger.finalize("success")
trainer.logger.save()
trainer.test(model=model, datamodule=dataloader)