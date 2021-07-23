#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.len

class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)
    def forward(self, x):
        return self.layer(x)
    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}
    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)
    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)
    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)
    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=2)
    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=2)
    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=2)

def run():
    early_stopping = EarlyStopping(monitor="valid_loss")
    checkpoint_callback = ModelCheckpoint(dirpath="logs", monitor="valid_loss")

    cli = LightningCLI(
        BoringModel,
        seed_everything_default=123,
        trainer_defaults={
            "max_epochs": 2,
            "callbacks": [
                checkpoint_callback,
                early_stopping,
            ],
            "logger": {
                "class_path": "pytorch_lightning.loggers.TestTubeLogger",
                "init_args": {
                    "save_dir": "logs",
                    "create_git_tag": True,
                },
            },
        },
    )
    # cli.trainer.test(cli.model)

if __name__ == '__main__':
    run()