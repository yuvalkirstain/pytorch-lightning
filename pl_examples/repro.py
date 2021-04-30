import os
from argparse import ArgumentParser

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.lr_scale = kwargs.get("lr_scale")
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss, sync_dist=False)
        assert isinstance(self.trainer.model, DistributedDataParallel)
        assert isinstance(self.trainer.training_type_plugin, DDPPlugin)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss, sync_dist=False)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss, sync_dist=False)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=(0.1 * self.lr_scale))


def run():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--name", type=str, default="debug")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.set_defaults(
        max_epochs=5,
        log_every_n_steps=1,
        limit_val_batches=0,
    )
    args = parser.parse_args()
    logger = WandbLogger(project="ddp-parity-1.3.0", name=args.name)
    trainer = Trainer.from_argparse_args(args, logger=logger, plugins=[DDPPlugin(find_unused_parameters=False)])
    model = BoringModel(**vars(args))
    train_data = DataLoader(RandomDataset(32, 6400), batch_size=args.batch_size)
    val_data = DataLoader(RandomDataset(32, 6400), batch_size=args.batch_size)
    test_data = DataLoader(RandomDataset(32, 6400), batch_size=args.batch_size)
    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)


if __name__ == '__main__':
    seed_everything(0, workers=True)
    run()
