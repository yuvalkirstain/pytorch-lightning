import os

import torch
from sparseml.pytorch.optim import ScheduledModifierManager
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import Callback, LightningModule, Trainer


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


class SparseMLCallback(Callback):

    def __init__(self, recipe_path: str):
        self.recipe_path = recipe_path
        self.manager = ScheduledModifierManager.from_yaml(self.recipe_path)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        steps_per_epoch = self.num_training_steps_per_epoch(trainer)
        trainer.optimizers = [
            self.manager.modify(pl_module, optimizer, steps_per_epoch, epoch=0) for optimizer in trainer.optimizers
        ]

    def num_training_steps_per_epoch(self, trainer: Trainer) -> int:
        """Total training steps inferred from datamodule and devices."""
        if isinstance(trainer.limit_train_batches, int) and trainer.limit_train_batches != 0:
            dataset_size = trainer.limit_train_batches
        elif isinstance(trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * trainer.limit_train_batches)
        else:
            dataset_size = len(trainer.datamodule.train_dataloader())

        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)

        effective_batch_size = trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size)

        if trainer.max_steps and trainer.max_steps < max_estimated_steps:
            return trainer.max_steps
        return max_estimated_steps

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.manager.finalize(pl_module)


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        weights_summary=None,
        callbacks=SparseMLCallback('config.yaml')
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
    trainer.test(model, dataloaders=test_data)


if __name__ == '__main__':
    run()
