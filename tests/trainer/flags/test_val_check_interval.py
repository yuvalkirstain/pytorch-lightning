# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from tests.helpers import BoringModel


@pytest.mark.parametrize("max_epochs", [1, 2, 3])
@pytest.mark.parametrize("denominator", [1, 3, 4])
def test_val_check_interval(tmpdir, max_epochs, denominator):
    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.train_epoch_calls = 0
            self.val_epoch_calls = 0

        def on_train_epoch_start(self) -> None:
            self.train_epoch_calls += 1

        def on_validation_epoch_start(self) -> None:
            if not self.trainer.sanity_checking:
                self.val_epoch_calls += 1

    model = TestModel()
    trainer = Trainer(max_epochs=max_epochs, val_check_interval=1 / denominator, logger=False)
    trainer.fit(model)

    assert model.train_epoch_calls == max_epochs
    assert model.val_epoch_calls == max_epochs * denominator


def test_val_check_interval_with_steps(tmpdir):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            print(f"batch idx = {batch_idx}")
            return super().training_step(batch, batch_idx)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=64, val_check_interval=8)
    trainer.fit(model)


def test_val_check_interval_user(tmpdir):
    class RandomDataset(torch.utils.data.Dataset):
        def __init__(self, data_dim, label_dim, length):
            super().__init__()
            self.data_dim = data_dim
            self.label_dim = label_dim
            self.length = length

        def __getitem__(self, idx):
            return torch.randn(*self.data_dim), torch.randn(*self.label_dim)

        def __len__(self):
            return self.length

    class RandomDataModule(pl.LightningDataModule):
        def __init__(self, data_dim, label_dim, length):
            super().__init__()
            self.data_dim = data_dim
            self.label_dim = label_dim
            self.length = length

        def setup(self, stage=None):
            self.dataset = RandomDataset(self.data_dim, self.label_dim, self.length)

        def train_dataloader(self):
            loader = torch.utils.data.DataLoader(self.dataset, batch_size=10, num_workers=0, drop_last=True)
            return loader

        def val_dataloader(self):
            return torch.utils.data.DataLoader(self.dataset, batch_size=10, num_workers=0, drop_last=True)

    class SimpleLinear(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(in_features=3, out_features=1)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters())

        def training_step(self, batch, batch_idx):
            print(f"batch idx = {batch_idx}")
            data, label = batch
            pred = self.layer(data)
            return torch.nn.functional.mse_loss(pred, label)

        def validation_step(self, batch, _batch_idx):
            data, label = batch
            pred = self.layer(data)
            return torch.nn.functional.mse_loss(pred, label)

    random_datamodule = RandomDataModule((3,), (1,), 1000)
    model = SimpleLinear()
    trainer = pl.Trainer(max_steps=500, val_check_interval=5)
    trainer.fit(model, random_datamodule)
