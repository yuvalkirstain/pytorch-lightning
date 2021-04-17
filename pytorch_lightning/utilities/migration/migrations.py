from functools import wraps
from typing import Optional

import torch
import logging

log = logging.getLogger(__name__)


class Migration:

    def __init__(self, target: str, requires: Optional[str]):
        self.target = target
        self.requires = requires

    def __call__(self, fn: callable) -> callable:
        @wraps(fn)
        def wrapper(ckpt):
            if ckpt["pytorch-lightning_version"] != self.requires:
                log.error("skipping")
                return ckpt
            new_ckpt = fn(ckpt)
            new_ckpt["pytorch-lightning_version"] = self.target
        return wrapper


@Migration(target="1.2.7", requires="1.2.6")
def upgrade_something_else(checkpoint: dict) -> dict:
    print(checkpoint)


@Migration(target="1.2.8", requires="1.2.7")
def upgrade_callback_names(checkpoint: dict) -> dict:
    print(checkpoint)



migration_chain = [
    upgrade_callback_names,
]

if __name__ == "__main__":
    checkpoint = torch.load("example.ckpt")
    upgrade_callback_names(checkpoint)

