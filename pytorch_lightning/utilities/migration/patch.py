import pytorch_lightning.utilities.argparse


class pl_legacy_patch:
    """
    Registers legacy artifacts (classes, methods, etc.) that were removed but still need to be
    included for unpickling old checkpoints.
    """

    def __enter__(self):
        setattr(pytorch_lightning.utilities.argparse, "_gpus_arg_default", lambda x: x)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        delattr(pytorch_lightning.utilities.argparse, "_gpus_arg_default")
