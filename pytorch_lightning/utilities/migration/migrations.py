import torch

from pytorch_lightning.utilities.migration.base import Migration, upgrade_checkpoint
from pytorch_lightning.utilities.migration.patch import pl_legacy_patch


@Migration(target="1.2.8")
def upgrade_something_else(checkpoint: dict) -> dict:
    return checkpoint


@Migration(target="1.2.9")
def upgrade_callback_state_identifiers(checkpoint):
    if "callbacks" not in checkpoint:
        return
    callbacks = checkpoint["callbacks"]
    checkpoint["callbacks"] = dict((callback_type.__name__, state) for callback_type, state in callbacks.items())
    return checkpoint


if __name__ == "__main__":
    with pl_legacy_patch():
        ckpt = torch.load("gpus-default-legacy.ckpt")
        # checkpoint = torch.load("example.ckpt")
        # checkpoint["pytorch-lightning_version"] = "1.2.6"

    ckpt = upgrade_checkpoint(ckpt)
    from pprint import pprint
    pprint(ckpt)
