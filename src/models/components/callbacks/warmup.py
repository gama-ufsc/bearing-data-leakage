from lightning.pytorch.callbacks import Callback


class WarmupSuppressLRonPlateauCallback(Callback):
    def __init__(self, warmup_epochs: int = 5):
        self.warmup_epochs = warmup_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        # Only apply scheduler step if it's ReduceLROnPlateau and we're past warmup
        for scheduler in trainer.lr_scheduler_configs:
            if scheduler.reduce_on_plateau and current_epoch < self.warmup_epochs:
                scheduler.should_update = False
            else:
                scheduler.should_update = True
