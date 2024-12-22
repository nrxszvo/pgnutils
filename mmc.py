from typing import Dict, Optional
from dataclasses import dataclass

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)

from model import ModelArgs, Transformer


@dataclass
class MMCModuleArgs:
    name: str
    outdir: str
    model_args: ModelArgs
    opening_moves: int
    NOOP: int
    lr_scheduler_params: Dict
    max_steps: int
    val_check_steps: int
    random_seed: Optional[int]
    strategy: Optional[str]
    devices: Optional[int]


class MimicChessModule(L.LightningModule):
    def __init__(self, params: MMCModuleArgs):
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        L.seed_everything(params.random_seed, workers=True)
        self.model = None
        self.model_args = params.model_args
        self.opening_moves = params.opening_moves
        self.NOOP = params.NOOP
        self.val_check_steps = params.val_check_steps
        self.max_steps = params.max_steps
        self.loss = F.cross_entropy
        if params.name:
            logger = TensorBoardLogger(".", name="L", version=params.name)
        else:
            logger = None
        val_check_interval = min(params.val_check_steps, params.max_steps)
        self.lr_scheduler_params = params.lr_scheduler_params
        if torch.cuda.is_available():
            precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else 16
        else:
            precision = 32

        self.trainer_kwargs = {
            "logger": logger,
            "max_steps": params.max_steps,
            "val_check_interval": val_check_interval,
            "check_val_every_n_epoch": None,
            "strategy": params.strategy,
            "devices": params.devices,
            "precision": precision,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        }
        self.callbacks = [
            TQDMProgressBar(),
        ]
        if params.outdir is not None:
            self.callbacks.append(
                ModelCheckpoint(
                    dirpath=params.outdir,
                    filename=params.name + "-{train_loss:.2f}",
                    every_n_train_steps=val_check_interval,
                )
            )

        self._init_model()
        self.trainer = L.Trainer(callbacks=self.callbacks, **self.trainer_kwargs)

    def _init_model(self):
        if self.model is not None:
            return
        self.model = Transformer(self.model_args)

    def num_params(self):
        nparams = 0
        nwflops = 0
        for name, w in self.model.named_parameters():
            if w.requires_grad:
                nparams += w.numel()
                if (
                    "embeddings" not in name
                    and "norm" not in name
                    and "bias" not in name
                ):
                    nwflops += w.numel()
        return nparams, nwflops

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
        elif name == "Cosine":
            min_lr = self.lr_scheduler_params["min_lr"]
            scheduler = CosineAnnealingLR(
                optimizer=optimizer, T_max=self.max_steps, eta_min=min_lr
            )
            freq = 1
        elif name == "WarmUpCosine":
            warmup_steps = self.lr_scheduler_params["warmup_steps"]
            warmupLR = LinearLR(
                optimizer=optimizer,
                start_factor=1 / warmup_steps,
                end_factor=1,
                total_iters=warmup_steps,
            )
            min_lr = self.lr_scheduler_params["min_lr"]
            cosineLR = CosineAnnealingLR(
                optimizer=optimizer, T_max=self.max_steps - warmup_steps, eta_min=min_lr
            )
            scheduler = SequentialLR(
                optimizer=optimizer,
                schedulers=[warmupLR, cosineLR],
                milestones=[warmup_steps],
            )
            freq = 1
        config = {
            "scheduler": scheduler,
            "frequency": freq,
            "interval": "step",
            "monitor": "valid_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": config}

    def forward(self, tokens):
        return self.model(tokens)

    def _chomp_pred(self, pred):
        pred = pred.permute(0, 2, 1)
        return pred[:, :, self.opening_moves - 1 :]

    def _get_loss(self, logits, target):
        logits = self._chomp_pred(logits)
        return self.loss(logits, target, ignore_index=self.NOOP)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input"])
        loss = self._get_loss(logits, batch["target"])
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input"])
        valid_loss = self._get_loss(logits, batch["target"])

        if torch.isnan(valid_loss):
            raise Exception("Loss is NaN, training stopped.")

        self.log("valid_loss", valid_loss, prog_bar=True, sync_dist=True)
        return valid_loss

    def sample_top_n(self, probs, n):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        return probs_idx[:, :, :n], probs_sort[:, :, :n]

    def sample_top_p(self, probs, p, tgt):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        bs, seqlen, nclass = probs_sort.shape
        next_token = torch.multinomial(
            probs_sort.reshape(-1, nclass), num_samples=1
        ).reshape(bs, seqlen, 1)
        next_token = torch.gather(probs_idx, -1, next_token)
        next_token[tgt == self.NOOP] = self.NOOP
        return next_token

    def predict_step(self, batch, batch_idx):
        logits = self(batch["input"])
        logits = self._chomp_pred(logits)
        probs = torch.softmax(logits, dim=1)

        sprobs, sgrps = torch.sort(probs, dim=1, descending=True)
        sprobs = sprobs[:, :5]
        sgrps = sgrps[:, :5]

        tgts = batch["target"]
        wtgt = tgts[:, 0]
        btgt = tgts[:, 1]
        _, nclass, _ = probs.shape

        tprobs = torch.empty_like(probs)
        adjprobs = torch.empty_like(probs)
        for i, tgt in enumerate([wtgt, btgt]):
            mask = F.one_hot(tgt, nclass).unsqueeze(-1)
            lo = F.one_hot(torch.clamp(tgt - 1, min=0), nclass).unsqueeze(-1)
            hi = F.one_hot(torch.clamp(tgt + 1, max=nclass - 1), nclass).unsqueeze(-1)
            tprobs[:, :, i::2] = probs[:, :, i::2] * mask
            adjprobs[:, :, i::2] = probs[:, :, i::2] * (mask + lo + hi)
        tprobs = tprobs.sum(dim=1)
        adjprobs = adjprobs.sum(dim=1)

        return {
            "sorted_probs": sprobs,
            "sorted_groups": sgrps,
            "target_probs": tprobs,
            "adjacent_probs": adjprobs,
            "target_groups": tgts,
        }

    def fit(self, datamodule, ckpt=None):
        self.trainer.fit(self, datamodule=datamodule, ckpt_path=ckpt)
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())

    def predict(self, datamodule):
        tkargs = self.trainer_kwargs
        trainer = L.Trainer(callbacks=[TQDMProgressBar()], **tkargs)
        outputs = trainer.predict(self, datamodule)
        return outputs
