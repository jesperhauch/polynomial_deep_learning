from models.neural_nets import *
from models.pi_nets import *
from utils.model_utils import *
from utils.logging_helper import run_information
from data.polynomials import *
from data.optimization_functions import *
from data.epidemiology import *
from lightning import Trainer
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
parser.add_argument("--data_std", type=float, default=1.0)
parser.add_argument("--noise", action="store_true")
parser.add_argument("--standardize", action="store_true")
parser.add_argument("--n_data", type=int, default=100000)
parser.add_argument("--batch_size", type=int, default=32)

# Model arguments
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--n_neurons", type=int, default=64)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--out_dim", type=int, default=1)

# Trainer/logger arguments
parser.add_argument("--epochs", type=int, default=30)

# Parse arguments
args = parser.parse_args()

# Initialize dataloader and create relevant logging
if args.data_generator == "Normal":
    polynomial = eval(args.polynomial, globals())
    in_dim = polynomial.__code__.co_argcount
    torch.manual_seed(0)
    data_gen = NormalGenerator(polynomial, in_dim, args.n_data, args.data_mean, args.data_std, args.noise)
    data_args = {"data_dist": f"Normal({str(args.data_mean)},{str(args.data_std)})",
                 "noise": str(args.noise),
                 "polynomial_name": args.polynomial_name}
    log_name = "Polynomials/" + args.polynomial_name
else: 
    try:
        data_gen = eval(args.data_generator, globals())(args.n_data, args.noise, args.standardize)
        in_dim = data_gen.n_features()
    except:
        raise NotImplementedError("The generator you are looking for is not implemented.")
    data_args = {"noise": args.noise}
    log_name = f"Optimization/{type(data_gen).__name__}"

dataloader = PolynomialModule(fn_data=data_gen, batch_size=args.batch_size)
log_name += "_noise/" if args.noise else "/"

# Choose model
model_args = {"input_size": in_dim,
              "hidden_sizes": [args.n_neurons]*args.n_layers,
              "hidden_size": args.n_neurons,
              "output_size": args.out_dim}
try:
    model = eval(args.model, globals())(**model_args)
except:
    raise NotImplementedError("The model {n} is not implemented or imported correctly.")

# Initialize logger and Trainer
log_name += type(model).__name__
logger = TensorBoardLogger("tb_logs", name=log_name)
trainer = Trainer(limit_train_batches=64,max_epochs=args.epochs, log_every_n_steps=25, logger=logger)

# Log data hyperparams and output run information
run_information(logger, data_gen, model, **data_args)

# Start training
trainer.fit(model=model, datamodule=dataloader)
logger.finalize("success")