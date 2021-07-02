import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer, seed_everything


class RandomDataset(Dataset):

    def __getitem__(self, index):
        # 0.{random digits}
        # 1.{random digits}
        # 2.{random digits}
        # ...
        return torch.rand(1) + index

    def __len__(self):
        return 64


class RandomLightningModule(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 2)
        self.recorded_samples = []

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        # print(batch_idx, batch)
        self.recorded_samples.append(batch)
        return self(batch).sum()

    def train_dataloader(self):
        dataset = RandomDataset()
        dataloader = DataLoader(dataset, batch_size=2)
        return dataloader

    def configure_optimizers(self):
        return Adam(self.parameters())


def test_fastforward_sampler_and_dataset(tmpdir):
    print("initial training")
    seed_everything(1)
    model = RandomLightningModule()
    trainer = Trainer(max_steps=3, progress_bar_refresh_rate=0, weights_summary=None)
    trainer.fit(model)

    print(torch.cat(model.recorded_samples))
    indices = [int(x) for x in torch.cat(model.recorded_samples).floor()]
    assert indices == [0, 1, 2, 3, 4, 5]

    ckpt_file = os.path.join(tmpdir, "one.ckpt")
    trainer.save_checkpoint(ckpt_file)

    print("resuming")
    seed_everything(1)
    model = RandomLightningModule()
    trainer = Trainer(max_steps=6, progress_bar_refresh_rate=0, weights_summary=None, resume_from_checkpoint=ckpt_file)
    trainer.fit(model)

    print(torch.cat(model.recorded_samples))
    indices = [int(x) for x in torch.cat(model.recorded_samples).floor()]
    assert indices == [6, 7, 8, 9]
