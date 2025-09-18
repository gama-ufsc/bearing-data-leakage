import lightning as L
from typing import Optional


class CustomCheckpoint(L.pytorch.callbacks.ModelCheckpoint):
    """
    Custom Checkpoint callback that allows for manual specification of the checkpoint path.

    Inspired by: https://stackoverflow.com/questions/76148710/how-to-manually-specify-checkpoint-path-in-pytorchlightning#:~:text=0-,Problem,-The%20above%20answer
    """

    def __init__(self, dirpath, filename: Optional[str] = None, **kwargs):
        self.file_name = filename
        # self.num_ckpts = 0 # Used when multiple checkpoints per epoch are saved
        super().__init__(dirpath=dirpath, **kwargs)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(
            trainer=trainer, pl_module=pl_module, checkpoint=checkpoint
        )

        # self.num_ckpts += 1
        # self.file_name = "ckpt_" + f"{self.num_ckpts}".zfill(3) + "_{epoch}_{val_loss:.2f}"  # Update filename for next checkpoint
        trainer.checkpoint_callback.filename = (
            self.file_name
        )  # Money line! this is where the update gets applied
