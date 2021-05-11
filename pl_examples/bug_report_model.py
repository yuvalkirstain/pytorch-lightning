import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer
import horovod.torch as hvd


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


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    # hvd.init()
    
    model = BoringModel()

    # def _filter_named_parameters(model, optimizer):
    #     opt_params = set([p for group in optimizer.param_groups for p in group.get("params", [])])
    #     return [(name, p) for name, p in model.named_parameters() if p in opt_params]
    #
    # optim = model.configure_optimizers()
    # hvd.DistributedOptimizer(
    #     optimizer=optim,
    #     named_parameters=_filter_named_parameters(model, optim),
    #     compression=hvd.Compression.none,
    # )


    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=10,
        limit_val_batches=10,
        max_epochs=5,
        weights_summary=None,
        accelerator="horovod",
        gpus=1,
    )
    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)
    trainer.test(model, test_dataloaders=test_data)


if __name__ == '__main__':
    run()
