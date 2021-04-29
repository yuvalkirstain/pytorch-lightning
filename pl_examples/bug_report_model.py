import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import torchvision
from torchvision.datasets import STL10
from torchvision import transforms
import argparse

from pytorch_lightning.loggers import WandbLogger

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument("--working_dir", default="./", type=str)
parser.add_argument("--dataset_dir", default="./", type=str)
parser.add_argument("--gpus", default=[0], type=int, nargs="+")
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--name", default="debug", type=str)

args = parser.parse_known_args()[0]


class DataModule(LightningDataModule):

    def __init__(self, train_dataset):
        super().__init__()
        self.train_dataset = train_dataset
        self.batch_size = args.batch_size // len(args.gpus)
        self.num_workers = args.num_workers // 2

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )


def get_datamodule(dataset_dir):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    augmentation = [
        transforms.RandomResizedCrop(96, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ]
    dataset = STL10(dataset_dir, 'test', download=True, transform=transforms.Compose(augmentation))
    return DataModule(dataset)


class ToyModel(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.resnet1 = torchvision.models.resnet18()
        self.resnet1.fc = nn.Linear(512, 10)
        self.resnet2 = torchvision.models.resnet18()
        self.resnet2.fc = nn.Linear(512, 10)
        self.Loss1 = nn.CrossEntropyLoss()
        self.Loss2 = nn.CrossEntropyLoss()

    def configure_optimizers(self):

        prms1 = [{'params': self.resnet1.parameters(), 'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9}]
        optimizer_1 = torch.optim.SGD(prms1)

        prms2 = [{'params': self.resnet2.parameters(), 'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9}]
        optimizer_2 = torch.optim.SGD(prms2)

        return [optimizer_1]

    def forward(self, x, isfirst=True):
        if isfirst:
            return self.resnet1(x)
        else:
            return self.resnet2(x)

    def training_step(self, batch, batch_idx):
        optimizer_idx = 0
        x, y = batch

        if optimizer_idx == 0:
            pr1 = self(x, True)
            loss1 = self.Loss1(pr1, y)
            self.log('loss1', loss1, sync_dist=True, prog_bar=True, on_step=True, on_epoch=True)
            return loss1
        else:
            pr2 = self(x, False)
            loss2 = self.Loss2(pr2, y)
            self.log('loss2', loss2, sync_dist=True, prog_bar=True, on_step=True, on_epoch=True)
            return loss2


def main():
    pl.seed_everything(0)
    data_module = get_datamodule(args.dataset_dir)

    model = ToyModel(**vars(args))

    logger = WandbLogger(project="ddp-parity-1.3.0", name=args.name)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        precision=16,
        deterministic=True,
        default_root_dir=args.working_dir,
        accelerator='ddp',
        sync_batchnorm=True,
        logger=logger,
    )

    trainer.fit(model, data_module)

    return model.float()


if __name__ == '__main__':
    model = main()
