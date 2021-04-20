import torch

from pytorch_lightning.utilities.migration.base import Migration, get_version
from pytorch_lightning.utilities.migration.patch import pl_legacy_patch


@Migration(requires="1.2.7")
def upgrade_callback_names(checkpoint: dict) -> dict:
    if "callbacks" not in checkpoint:
        return checkpoint
    checkpoint["callbacks"] = reversed(checkpoint["callbacks"])
    print(get_version(checkpoint))
    return checkpoint


@Migration(requires="1.2.8")
def upgrade_something_else(checkpoint: dict) -> dict:
    return checkpoint


if __name__ == "__main__":
    with pl_legacy_patch():
        checkpoint = torch.load("gpus-default-legacy.ckpt")
        # checkpoint = torch.load("example.ckpt")
        # checkpoint["pytorch-lightning_version"] = "1.2.6"
    # getattr()
    checkpoint = Migration.migrate(checkpoint)
    print(checkpoint)
