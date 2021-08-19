from pytorch_lightning.utilities import rank_zero_deprecation


class SLURMConnector:
    def __init__(self, trainer):
        self.trainer = trainer
        _slurm_connector_deprecation_warning()

    def register_slurm_signal_handlers(self):  # pragma: no-cover
        _slurm_connector_deprecation_warning()

    def sig_handler(self, signum, frame):  # pragma: no-cover
        _slurm_connector_deprecation_warning()

    def term_handler(self, signum, frame):  # pragma: no-cover
        _slurm_connector_deprecation_warning()


def _slurm_connector_deprecation_warning():
    rank_zero_deprecation(
        "`SLURMConnector` is deprecated in 1.5 and will be removed in 1.7. Attaching signal handlers was moved"
        " to the `SLURMEnvironment` and can be overridden there."
    )
