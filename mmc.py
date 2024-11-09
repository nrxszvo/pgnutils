import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from model import Transformer, ModelArgs

from typing import Optional, Dict


class MimicChessModule(L.LightningModule):
    def __init__(
        self,
        name: str,
        outdir: str,
        model_args: ModelArgs,
        min_moves: int,
        NOOP: int,
        lr_scheduler_params: Dict,
        max_steps: Optional[int] = 100,
        val_check_steps: Optional[int] = 100,
        random_seed: Optional[int] = 0,
        strategy: Optional[str] = "auto",
        devices: Optional[int] = 1,
    ):
        super().__init__()
        L.seed_everything(random_seed, workers=True)
        self.model = None
        self.min_moves = min_moves
        self.NOOP = NOOP
        self.val_check_steps = val_check_steps
        self.max_steps = max_steps
        self.loss = F.cross_entropy
        logger = TensorBoardLogger(".", name="L", version=name)
        val_check_interval = min(val_check_steps, max_steps)
        self.lr_scheduler_params = lr_scheduler_params
        self.trainer_kwargs = {
            "logger": logger,
            "max_steps": max_steps,
            "val_check_interval": val_check_interval,
            "check_val_every_n_epoch": None,
            "strategy": strategy,
            "devices": devices,
        }
        self.callbacks = [
            TQDMProgressBar(),
            ModelCheckpoint(
                dirpath=outdir,
                filename=name,
                save_weights_only=True,
                every_n_train_steps=val_check_interval,
            ),
        ]

        if self.model is not None:
            return
        self.model = Transformer(model_args)

        print(f"# model params: {self.num_params():.2e}")
        self.trainer = L.Trainer(callbacks=self.callbacks, **self.trainer_kwargs)

    def num_params(self):
        nparams = 0
        for w in self.model.parameters():
            if w.requires_grad:
                nparams += w.numel()
        return nparams

    def configure_optimizers(self):
        lr = self.lr_scheduler_params["lr"]
        optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=lr)
        name = self.lr_scheduler_params["name"]
        if name == "ReduceLROnPlateau":
            gamma = self.lr_scheduler_params["gamma"]
            threshold = self.lr_scheduler_params["threshold"]
            patience = self.lr_scheduler_params["patience"]
            min_lr = self.lr_scheduler_params["min_lr"]
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                factor=gamma,
                threshold=threshold,
                patience=patience,
                min_lr=min_lr,
            )
            freq = self.val_check_steps
        elif name == "StepLR":
            gamma = self.lr_scheduler_params["gamma"]
            num_decay = self.lr_scheduler_params["num_lr_decays"]
            step_size = int(self.max_steps / num_decay)
            scheduler = StepLR(
                optimizer=optimizer,
                step_size=step_size,
                gamma=gamma,
            )
            freq = 1
        config = {
            "scheduler": scheduler,
            "frequency": freq,
            "interval": "step",
            "monitor": "valid_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": config}

    def forward(self, inputs, target=None):
        insample_y = inputs[0]
        return self.model(insample_y, 0)

    def training_step(self, batch, batch_idx):
        logits = self([batch["input"]])
        logits = logits[:, self.min_moves :].permute(0, 2, 1)
        tgt = batch["target"]
        loss = self.loss(logits, tgt, ignore_index=self.NOOP)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self([batch["input"]])
        logits = logits[:, self.min_moves :].permute(0, 2, 1)
        tgt = batch["target"]
        valid_loss = self.loss(logits, tgt, ignore_index=self.NOOP)

        if torch.isnan(valid_loss):
            raise Exception("Loss is NaN, training stopped.")

        self.log("valid_loss", valid_loss, prog_bar=True, sync_dist=True)
        return valid_loss

    def predict_step(self, batch, batch_idx):
        return self([batch["input"]])

    def fit(self, datamodule):
        self.trainer.fit(self, datamodule=datamodule)
        print(torch.cuda.memory_summary())

    def predict(self, datamodule):
        tkargs = self.trainer_kwargs
        tkargs["strategy"] = "auto"
        tkargs["devices"] = 1
        trainer = L.Trainer(callbacks=[TQDMProgressBar()], **tkargs)
        pred = trainer.predict(self, datamodule)

        y_true = datamodule.testset.series[:, self.input_size :]
        nseries, npts, ndim = y_true.shape
        # unfold = torch.nn.Unfold((self.h, 1))
        # y_true = unfold(yt_raw.permute(0, 2, 1).unsqueeze(-1)).permute(0, 2, 1)
        # y_true = y_true.reshape(nseries, -1, ndim, self.h).permute(0, 1, 3, 2)
        # y_true = y_true[:, :: self.stride]

        y_hat = torch.cat(pred, dim=0)
        y_hat = y_hat.reshape(nseries, -1, self.h, ndim)

        return y_hat.numpy(), y_true

    def validate(self, datamodule):
        tkargs = self.trainer_kwargs
        tkargs["strategy"] = "auto"
        tkargs["devices"] = 1
        trainer = L.Trainer(callbacks=[TQDMProgressBar()], **tkargs)
        res = trainer.validate(self, datamodule)
        return res

    def on_save_checkpoint(self, checkpoint):
        """
        Tentative fix for FSDP checkpointing issue
        """
        if not checkpoint.get("state_dict", None):
            state_dict = self.trainer.model.state_dict()
            checkpoint["state_dict"] = state_dict
        return super().on_save_checkpoint(checkpoint)
