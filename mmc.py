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

from model import ModelArgs, Transformer, EloClassifier


@dataclass
class MMCModuleArgs:
    name: str
    outdir: str
    model_args: ModelArgs
    min_moves: int
    NOOP: int
    lr_scheduler_params: Dict
    max_steps: int
    val_check_steps: int
    random_seed: Optional[int]
    strategy: Optional[str]
    devices: Optional[int]
    classifier: bool


class MimicChessCoreModule(L.LightningModule):
    def __init__(self, params: MMCModuleArgs):
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        L.seed_everything(params.random_seed, workers=True)
        self.model = None
        self.model_args = params.model_args
        self.min_moves = params.min_moves
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
            ModelCheckpoint(
                dirpath=params.outdir,
                filename=params.name + "-{train_loss:.2f}",
                every_n_train_steps=val_check_interval,
            ),
        ]

        self._init_model()
        self.trainer = L.Trainer(callbacks=self.callbacks, **self.trainer_kwargs)

    def _init_model(self):
        if self.model is not None:
            return
        if self.params.classifier:
            self.model = EloClassifier(self.model_args)
        else:
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

    def forward(self, tokens, last_idx):
        return self.model(tokens, last_idx)

    def _chomp_logits(self, logits):
        logits = logits.permute(0, 2, 1)
        logits = logits[:, :, self.min_moves - 1 :]
        return logits

    def _separate_logits(self, logits, batch):
        logits = self._chomp_logits(logits)
        wlogits = torch.index_select(logits, 0, batch["offset_heads"][:, 0])
        blogits = torch.index_select(logits, 0, batch["offset_heads"][:, 1])
        wtgt = batch["w_target"]
        btgt = batch["b_target"]
        return wlogits, blogits, wtgt, btgt

    def _get_loss(self, logits, batch):
        nhead = self.params.model_args.n_elo_heads
        if self.params.classifier:
            wloss = self.loss(logits[:, :nhead], batch["heads"][:, 0])
            bloss = self.loss(logits[:, nhead:], batch["heads"][:, 1])
        else:
            wlogits, blogits, wtgt, btgt = self._separate_logits(logits, batch)
            wloss = self.loss(wlogits, wtgt, ignore_index=self.NOOP)
            bloss = self.loss(blogits, btgt, ignore_index=self.NOOP)

        return (wloss + bloss) / 2

    def training_step(self, batch, batch_idx):
        logits = self(batch["input"], batch["last_idx"])
        loss = self._get_loss(logits, batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input"], batch["last_idx"])
        valid_loss = self._get_loss(logits, batch)

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
        logits = self._chomp_logits(logits)
        probs = torch.softmax(logits, dim=-2)
        sprobs, stokens = torch.sort(probs, dim=-2, descending=True)
        sprobs = sprobs[:, :5]
        stokens = stokens[:, :5]

        tgt = batch["w_target"]
        tgt[:, 1::2] = batch["b_target"][:, 1::2]

        bs, seqlen = tgt.shape
        bsheads, nclass, _ = probs.shape
        nheads = bsheads // bs
        oh_tgt = tgt[:, None].repeat(1, nheads, 1).reshape(-1, seqlen)
        mask = F.one_hot(oh_tgt, nclass).permute(0, 2, 1)
        tprobs = (probs * mask).sum(dim=1)

        return {
            "sorted_tokens": stokens,
            "sorted_probs": sprobs,
            "target_probs": tprobs,
            "offset_heads": batch["offset_heads"],
            "heads": batch["heads"],
            "opening": batch["opening"],
            "targets": tgt.unsqueeze(1),
        }

    def init_classifier(self, ckpt_fn):
        cf_dict = self.state_dict()
        tf_dict = torch.load(ckpt_fn, map_location="cpu")["state_dict"]
        tf_dict = {k: v for k, v in tf_dict.items() if k in cf_dict}
        cf_dict.update(tf_dict)
        self.load_state_dict(cf_dict)
        for name, param in self.named_parameters():
            if name.startswith("model.layers") or name in [
                "model.tok_embeddings.weight",
                "model.norm.weight",
            ]:
                param.requires_grad = False

    def fit(self, datamodule, ckpt=None):
        self.trainer.fit(self, datamodule=datamodule, ckpt_path=ckpt)
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())

    def predict(self, datamodule):
        tkargs = self.trainer_kwargs
        trainer = L.Trainer(callbacks=[TQDMProgressBar()], **tkargs)
        outputs = trainer.predict(self, datamodule)
        return outputs

    def on_save_checkpoint(self, checkpoint):
        """
        Tentative fix for FSDP checkpointing issue
        """
        if not checkpoint.get("state_dict", None):
            state_dict = self.trainer.model.state_dict()
            checkpoint["state_dict"] = state_dict
        return super().on_save_checkpoint(checkpoint)


class MimicChessHeadModule(MimicChessCoreModule):
    def __init__(self, params: MMCModuleArgs, core_ckpt: str):
        self.params = params
        self.core_ckpt = core_ckpt
        super().__init__(params)

    def _init_model(self):
        super()._init_model()
        # exclude_layer = f"layers.{self.params.model_args.n_layers-1}"
        # for name, param in self.model.named_parameters():
        #    if (
        #        name not in ["output.weight", "norm.weight"]
        #        and exclude_layer not in name
        #    ):
        #        param.requires_grad = False

        ckpt = torch.load(self.core_ckpt, map_location="cpu", weights_only=True)
        self.load_state_dict(ckpt["state_dict"], strict=True)
