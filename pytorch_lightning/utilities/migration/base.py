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
    "1.2.9",
    "1.3.0rc0",
    "1.3.0rc1",
]

if pytorch_lightning.__version__ not in version_history:
    version_history.append(pytorch_lightning.__version__)


def default_migration(checkpoint):
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


all_migrations = dict((ver, default_migration) for ver in version_history)


def get_version(checkpoint: dict) -> str:
    return checkpoint["pytorch-lightning_version"]


def set_version(checkpoint: dict, version: str):
    checkpoint["pytorch-lightning_version"] = version


class Migration:
    """ Decorator for a function that upgrades a checkpoint from one version to the next. """

    def __init__(self, target: Optional[str]):
        self.target_version = target

    def __call__(self, upgrade_fn: callable) -> callable:
        if getattr(upgrade_fn, "_migration_registered", False) and all_migrations[self.target_version] != default_migration:
            raise ValueError(
                f"Tried to register a new migration {upgrade_fn.__name__}, but"
                f" there is already a migration for version {self.target_version}:"
                f" {all_migrations[self.target_version].__name__}"
            )
        all_migrations[self.target_version] = upgrade_fn
        upgrade_fn._migration_registered = True
        return upgrade_fn


def upgrade_checkpoint(checkpoint: dict) -> dict:
    for migration in all_migrations.values():
        if migration is None:
            checkpoint = default_migration(checkpoint)
        checkpoint = migration(checkpoint)
    return checkpoint

