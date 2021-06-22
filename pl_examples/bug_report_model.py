import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        # self.trainer.profiler.describe()  # why was this here??
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = MNIST("./MNIST", download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

autoencoder = LitAutoEncoder()


settings = [
    # accumulation, max_steps
    (1, 1000),
    (4, 250),
    (8, 125),
    (8, 250),
    (8, 500),
    (8, 1000),
]

for accumulation, max_steps in settings:
    trainer = pl.Trainer(
        profiler="simple",
        accumulate_grad_batches=accumulation,
        log_every_n_steps=50,
        max_steps=max_steps,
        gpus=1,
    )
    trainer.profiler.dirpath = "lightning_logs"
    trainer.profiler.filename = f'fprofiler_accumulation_{accumulation}_{max_steps}'
    trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
