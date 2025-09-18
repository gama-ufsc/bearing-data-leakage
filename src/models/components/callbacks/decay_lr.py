import lightning as L


class StepDecayLRCallback(L.pytorch.callbacks.Callback):
    def __init__(self, step_size: int = 100, gamma: float = 0.5):
        """
        Args:
            step_size (int): Number of epochs between each LR decay.
            gamma (float): Multiplicative factor for LR decay.
        """
        self.step_size = step_size
        self.gamma = gamma
        print("StepDecayLRCallback initialized!")  # <-- does this print?

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch > 0 and epoch % self.step_size == 0:
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= self.gamma

                    # print(f"[Epoch {epoch}] Reducing LR to {param_group['lr']:.5e}")
