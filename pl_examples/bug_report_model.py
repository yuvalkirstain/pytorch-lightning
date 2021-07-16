import gc

import torch
from pytorch_lightning.utilities.memory import garbage_collection_cuda
import time


def get_model():
    from pytorch_lightning import LightningModule

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    class RandomDataset(Dataset):
        def __init__(self, size, num_samples):
            self.len = num_samples
            self.data = torch.randn(num_samples, size)

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return self.len

    class BoringModel(LightningModule):

        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(32, 2)
            self.batch_size = 4

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 10000),
                            batch_size=self.batch_size)

        def forward(self, x):
            return self.layer(x)

        def loss(self, batch, prediction):
            # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
            return nn.MSELoss()(prediction, torch.ones_like(prediction))

        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            return loss

        def training_step_end(self, training_step_outputs):
            return training_step_outputs

        def training_epoch_end(self, outputs) -> None:
            torch.stack([x["loss"] for x in outputs]).mean()

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    model = BoringModel()
    return model


def get_cudas():
    cudas = []

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if str(obj.device) != 'cpu':
                    cudas += [obj]
        except:
            pass
    return cudas


def run(precision=32, accumulate_grad_batches=1):
        model = get_model()
        import pytorch_lightning as pl

        trainer = pl.Trainer(
            gpus='0',
            accumulate_grad_batches=accumulate_grad_batches,
            log_every_n_steps=1,
            precision=precision,
            deterministic=True,
            max_steps=32,
            reload_dataloaders_every_epoch=True,
            auto_lr_find=False,
            replace_sampler_ddp=True,
            terminate_on_nan=False,
            auto_scale_batch_size=True,
            weights_summary=None,
            progress_bar_refresh_rate=0,
        )

        trainer.fit(model)


def collect_garbage():
    garbage_collection_cuda()
    time.sleep(5)
    torch.cuda.empty_cache()
    garbage_collection_cuda()
    gc.collect()


if __name__ == '__main__':
    print("first run")
    torch.cuda.reset_accumulated_memory_stats()
    run()
    collect_garbage()
    print("memory:", torch.cuda.memory_allocated(), "max:", torch.cuda.max_memory_allocated())
    print(get_cudas())  # returns []

    print("second run")
    torch.cuda.reset_accumulated_memory_stats()
    run()
    collect_garbage()
    print("memory:", torch.cuda.memory_allocated(), "max:", torch.cuda.max_memory_allocated())
