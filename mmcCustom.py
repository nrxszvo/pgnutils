import os
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim

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


def test(params: MMCModuleArgs, dm: MMCDataModule):
    rank = dist.get_rank()

    print(f"Start running basic DDP example on rank {rank}.")
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = Transformer(params.model_args).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = F.cross_entropy

    lr = params.lr_scheduler_params["lr"]
    optimizer = optim.Adam(ddp_model.parameters(), lr=lr)
    optimizer.zero_grad()

    dm.setup(stage="fit")
    for batch in dm.train_dataloader():
        logits = ddp_model(batch["input"])
        logits = logits[:, params.min_moves - 1 :].permute(0, 2, 1)
        tgt = batch["target"].to(device_id)
        loss_fn(logits, tgt, ignore_index=params.NOOP).backward()
        optimizer.step()
        break

    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")
