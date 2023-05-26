from models.neural_nets import *
from models.pi_nets import *
from models.baselines import *
from models.base_model import *
from utils.model_utils import *
from utils.logging_helper import run_information
from data.epidemiology import *
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

parser = ArgumentParser()

# Data arguments
parser.add_argument("--seq_len", type=int, default=1000)
parser.add_argument("--lag_size", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)

# Model arguments
parser.add_argument("--multiplication_net", type=str, default="CCP")
parser.add_argument("--n_neurons", type=int, default=64)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_degree", type=int, default=2)

# Trainer/logger arguments
parser.add_argument("--epochs", type=int, default=100)

# Parse arguments
args = parser.parse_args()

# Initialize dataloader and create relevant logging
dataloader = EpidemiologyModule(batch_size=args.batch_size, lag_size=args.lag_size)
log_name = type(dataloader).__name__ + "/"

input_size = 3 # susceptible  and infected
model_args = {"input_size": input_size,
              "n_layers": args.n_layers,
              "hidden_size": args.n_neurons,
              "n_degree": args.n_degree}
try:
    multiplication_net = eval(args.multiplication_net, globals())
    model = SIRModelWrapper(multiplication_net, **model_args)
except Exception as inst:
    print(inst)
    raise NotImplementedError("The model {n} is not implemented or imported correctly.")

# Initialize logger and Trainer
log_name += type(model).__name__ + "/" + multiplication_net.__name__
logger = TensorBoardLogger("tb_logs", name=log_name)
early_stop_callback = EarlyStopping(monitor="val_r2", mode="max")
trainer = Trainer(limit_train_batches=64, max_epochs=args.epochs, log_every_n_steps=25, logger=logger, callbacks=[early_stop_callback])

# Start training
trainer.fit(model=model, datamodule=dataloader)
trainer.test(model=model, datamodule=dataloader)