import os
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

from model import ModelArgs, Transformer
from mmcdataset import MMCDataModule


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


class MimicChessCoreModule:
    def __init__(self, params: MMCModuleArgs):
        self.params = params
        L.seed_everything(params.random_seed, workers=True)
        self.model = None
        self.model_args = params.model_args
        self.min_moves = params.min_moves
        self.NOOP = params.NOOP
        self.max_steps = params.max_steps
        self.loss = F.cross_entropy
        self.logger = TensorBoardLogger(".", name="L", version=params.name)
        self.val_check_interval = min(params.val_check_steps, params.max_steps)
        self.lr_scheduler_params = params.lr_scheduler_params

        rank = dist.get_rank()
        if torch.cuda.is_available():
            self.amp_type = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.half
            )
            device_id = rank % torch.cuda.device_count()
        else:
            self.amp_type = torch.half
            device_id = "cpu"

        self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self.model = Transformer(params.model_args).to(device_id)
        self.ddp_model = DDP(self.model, device_ids=[device_id])

    def configure_optimizers(self):
        lr = self.lr_scheduler_params["lr"]
        self.optimizer = torch.optim.Adam(self.ddp_model.parameters(), lr=lr)
        name = self.lr_scheduler_params["name"]
        if name == "Cosine":
            min_lr = self.lr_scheduler_params["min_lr"]
            self.scheduler = CosineAnnealingLR(
                optimizer=self.optimizer, T_max=self.max_steps, eta_min=min_lr
            )
        elif name == "WarmUpCosine":
            warmup_steps = self.lr_scheduler_params["warmup_steps"]
            warmupLR = LinearLR(
                optimizer=self.optimizer,
                start_factor=1 / warmup_steps,
                end_factor=1,
                total_iters=warmup_steps,
            )
            min_lr = self.lr_scheduler_params["min_lr"]
            cosineLR = CosineAnnealingLR(
                optimizer=self.optimizer, T_max=self.max_steps, eta_min=min_lr
            )
            self.scheduler = SequentialLR(
                optimizer=self.optimizer,
                schedulers=[warmupLR, cosineLR],
                milestones=[warmup_steps],
            )

    def fit(self, dm: MMCDataModule):
        dm.setup(stage="fit")
        for i, batch in enumerate(dm.train_dataloader()):
            if i == self.max_steps:
                break
            with torch.autograd(device_type=self.accelerator, dtype=self.amp_type):
                logits = self.ddp_model(batch["input"])
                logits = logits[:, self.params.min_moves - 1 :].permute(0, 2, 1)
                tgt = batch["target"].to(self.device_id)
                loss = self.loss_fn(logits, tgt, ignore_index=self.params.NOOP)
                print(f"Rank {self.rank} loss: {loss.item():.2f}", end="\r")
            self.logger.log("train_loss", loss.item())
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            if (i + 1) % self.val_check_interval:
                self.save_checkpoint()
                self.validate()

        dist.destroy_process_group()

    def validate(self):
        pass

    def save_checkpoint(self):
        pass
