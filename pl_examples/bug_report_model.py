import time

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def on_train_batch_start(self, *args, **kwargs):
        self._start = time.monotonic()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        # self.trainer.profiler.describe()  # why was this here??
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        delta = time.monotonic() - self._start
        self.log("time", delta, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = MNIST("./MNIST", download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

autoencoder = LitAutoEncoder()

logger = WandbLogger(project="accumulation-perf")

settings = [
    # accumulation, max_steps
    # (1, 10000),
    (4, 2500000),
    # (8, 1250),
    # (8, 2500),
    # (8, 5000),
    # (8, 10000),
]

for accumulation, max_steps in settings:
    trainer = pl.Trainer(
        logger=logger,
        profiler="simple",
        accumulate_grad_batches=accumulation,
        log_every_n_steps=50,
        max_steps=max_steps,
        gpus=1,
    )
    trainer.profiler.dirpath = "lightning_logs"
    trainer.profiler.filename = f'fprofiler_accumulation_{accumulation}_{max_steps}'
    trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
