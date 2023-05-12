from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from models.base_model import SIRModelWrapper
from data.epidemiology import EpidemiologyModule
import yaml
from yaml import Loader
model_name = "PDC"
logger = TensorBoardLogger("tb_logs", name=f"EpidemiologyModule\\TemporalModelWrapper\\{model_name}")
continue_log = f"tb_logs\\EpidemiologyModule\\TemporalModelWrapper\\{model_name}\\n_deg_min1_for_r_100"

checkpoint = continue_log + "\\checkpoints\\epoch=99-step=6400.ckpt"
hparams = continue_log + "\\hparams.yaml"
kwargs = yaml.load(open(hparams), Loader)
dataloader = EpidemiologyModule(**kwargs)
model = SIRModelWrapper.load_from_checkpoint(checkpoint)
trainer = Trainer(limit_train_batches=64, max_epochs=300, log_every_n_steps=25, logger=logger)
trainer.fit(model, datamodule=dataloader, ckpt_path=checkpoint)
trainer.test(model=model, datamodule=dataloader)