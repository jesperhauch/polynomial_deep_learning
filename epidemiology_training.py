from torch.nn import RNN, LSTM, GRU
from models.neural_nets import *
from models.pi_nets import *
from models.baselines import *
from models.base_model import TemporalModelWrapper
from utils.model_utils import *
from utils.logging_helper import run_information
from data.epidemiology import *
import pytorch_lightning as pl
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

parser = ArgumentParser()

# Data arguments
parser.add_argument("--seq_len", type=int, default=1000)
parser.add_argument("--lag_size", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--poly_features", action="store_true")

# Model arguments
parser.add_argument("--multiplication_net", type=str, required=True)
parser.add_argument("--n_neurons", type=int, default=64)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_degree", type=int, default=2)

# Trainer/logger arguments
parser.add_argument("--epochs", type=int, default=30)

# Parse arguments
args = parser.parse_args()

# Initialize dataloader and create relevant logging
dataloader = EpidemiologyModule(batch_size=args.batch_size, lag_size=args.lag_size, 
                                poly_features=args.poly_features, n_degree=args.n_degree)
log_name = f"Temporal/{type(dataloader).__name__}/"

# Choose model
if args.poly_features:
    input_size = number_of_features(args.n_degree) # if polynomial features are used for EpidemiologyModule
else:
    input_size = 2 # susceptible  and infected
model_args = {"input_size": input_size,
              "n_layers": args.n_layers,
              "hidden_size": args.n_neurons,
              "hidden_sizes": [args.n_neurons]*args.n_layers,
              "n_degree": args.n_degree}
try:
    multiplication_net = eval(args.multiplication_net, globals())
    model = TemporalModelWrapper(multiplication_net, **model_args)
except:
    raise NotImplementedError("The model {n} is not implemented or imported correctly.")

# Initialize logger and Trainer
log_name += multiplication_net.__name__
logger = pl.loggers.TensorBoardLogger("tb_logs", name=log_name)
trainer = pl.Trainer(limit_train_batches=64, max_epochs=args.epochs, log_every_n_steps=25, logger=logger)

# Start training
trainer.fit(model=model, datamodule=dataloader)
trainer.logger.finalize("success")
trainer.logger.save()
trainer.test(model=model, datamodule=dataloader)