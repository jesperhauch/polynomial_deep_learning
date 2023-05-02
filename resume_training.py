import pytorch_lightning as pl
from models.base_model import TemporalModelWrapper
from data.epidemiology import EpidemiologyModule
import yaml
from yaml import Loader
model_name = "PDC"
logger = pl.loggers.TensorBoardLogger("tb_logs", name=f"EpidemiologyModule\TemporalModelWrapper\{model_name}")
continue_log = f"tb_logs\EpidemiologyModule\TemporalModelWrapper\{model_name}\\n_deg_min1_for_r_100"

checkpoint = continue_log + "\checkpoints\epoch=99-step=6400.ckpt"
hparams = continue_log + "\hparams.yaml"
kwargs = yaml.load(open(hparams), Loader)
dataloader = EpidemiologyModule(**kwargs)
model = TemporalModelWrapper.load_from_checkpoint(checkpoint)
trainer = pl.Trainer(limit_train_batches=64, max_epochs=300, log_every_n_steps=25, logger=logger)
trainer.fit(model, datamodule=dataloader, ckpt_path=checkpoint)
trainer.logger.finalize("success")
trainer.logger.save()
trainer.test(model=model, datamodule=dataloader)