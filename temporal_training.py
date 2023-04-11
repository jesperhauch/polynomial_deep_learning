from models.neural_nets import *
from models.pi_nets import *
from models.temporal_models import *
from utils.model_utils import *
from utils.logging_helper import run_information
from data.polynomials import *
from data.simulation_functions import *
from data.epidemiology import *
import pytorch_lightning as pl
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

parser = ArgumentParser()

# Data arguments
parser.add_argument("--seq_len", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=32)

# Model arguments
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--n_neurons", type=int, default=64)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_degree", type=int, default=2)
parser.add_argument("--out_dim", type=int, default=1)

# Trainer/logger arguments
parser.add_argument("--epochs", type=int, default=30)

# Parse arguments
args = parser.parse_args()

# Initialize dataloader and create relevant logging
dataloader = SimulationModule(dataset=Epidemiology(1/80, 1/160, 1000, [], None), batch_size=args.batch_size)
log_name = f"Temporal/{type(dataloader).__name__}"

# Choose model
model_args = {"input_size": 1,
              "n_layers": args.n_layers,
              "hidden_size": args.n_neurons,
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

# Start training
trainer.fit(model=model, datamodule=dataloader)
trainer.logger.finalize("success")
trainer.logger.save()
trainer.test(model=model, datamodule=dataloader)