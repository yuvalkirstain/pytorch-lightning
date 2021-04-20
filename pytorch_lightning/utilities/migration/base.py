from functools import wraps
from typing import Optional
import pytorch_lightning
import logging

log = logging.getLogger(__name__)

version_history = [
    "0.0.1",
    "0.0.2",
    "0.1.1",
    "0.1.2",
    "0.10.0",
    "0.2",
    "0.2.1",
    "0.2.2",
    "0.2.3",
    "0.2.4",
    "0.2.5",
    "0.2.6",
    "0.3",
    "0.3.1",
    "0.3.2",
    "0.3.3",
    "0.3.4",
    "0.3.5",
    "0.3.6",
    "0.4.0",
    "0.4.1",
    "0.4.2",
    "0.4.3",
    "0.4.4",
    "0.4.5",
    "0.4.6",
    "0.4.7",
    "0.4.8",
    "0.4.9",
    "0.5.0",
    "0.5.1",
    "0.5.2",
    "0.5.2.1",
    "0.5.3",
    "0.5.3.1",
    "0.5.3.2",
    "0.6.0",
    "0.7.0",
    "0.7.1",
    "0.7.2",
    "0.7.3",
    "0.7.4",
    "0.7.5",
    "0.7.6",
    "0.8.0",
    "0.8.1",
    "0.8.2",
    "0.8.3",
    "0.8.4",
    "0.8.5",
    "0.9.0",
    "0.9.1rc1",
    "0.9.1rc2",
    "0.9.1rc3",
    "1.0.0",
    "1.0.1",
    "1.0.2",
    "1.0.3",
    "1.0.4",
    "1.0.4rc1",
    "1.0.5",
    "1.0.6",
    "1.0.7",
    "1.0.8",
    "1.1.0",
    "1.1.0rc1",
    "1.1.1",
    "1.1.2",
    "1.1.3",
    "1.1.4",
    "1.1.5",
    "1.1.6",
    "1.1.7",
    "1.1.8",
    "1.2.0",
    "1.2.0rc0",
    "1.2.1",
    "1.2.2",
    "1.2.3",
    "1.2.4",
    "1.2.5",
    "1.2.6",
    "1.2.7",
    "1.2.8",
    "1.3.0rc0",
    "1.3.0rc1",
    pytorch_lightning.__version__,
]


def default_upgrade_rule(checkpoint):
    """ Upgrades to the next version by only replacing the current version with the new one. """
    # TODO: find more elegant version for the if below
    current = get_version(checkpoint)
    if current in version_history:
        idx = version_history.index(current)
        if idx < len(version_history) - 1:
            # upgrade to the next version without changes
            set_version(checkpoint, version_history[idx + 1])
    else:
        set_version(checkpoint, pytorch_lightning.__version__)
    return checkpoint


def get_version(checkpoint: dict) -> str:
    return checkpoint["pytorch-lightning_version"]


def set_version(checkpoint: dict, version: str):
    checkpoint["pytorch-lightning_version"] = version


class Migration:
    """ Decorator for a function that upgrades a checkpoint from one version to the next. """

    all_migrations = dict.fromkeys(version_history, [default_upgrade_rule])

    def __init__(self, requires: Optional[str]):
        self.required_version = requires

    def __call__(self, fn: callable) -> callable:
        @wraps(fn)
        def wrapper(ckpt):
            current_version = get_version(ckpt)
            if self.required_version and current_version != self.required_version:
                log.error(f"skipping, {current_version}")
                return ckpt
            new_ckpt = fn(ckpt)
            return new_ckpt

        self.all_migrations[self.required_version].insert(0, wrapper)
        return wrapper

    @staticmethod
    def migrate(checkpoint: dict) -> dict:
        for version_migrations in Migration.all_migrations.values():
            for migration in version_migrations:
                checkpoint = migration(checkpoint)
        return checkpoint
