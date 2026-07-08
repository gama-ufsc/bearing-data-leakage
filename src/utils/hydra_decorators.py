"""
Utils decorators and functions from lightning-hydra-template.
"""

import gc
import logging
from typing import Any, Callable, Dict, Tuple

import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    Parameters:
    -----------
    task_func : Callable
        The task function to be wrapped.

    Returns
    -------
    Callable
        The wrapped task function.

    Example:
    --------
    ```
    @task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    Notes:
    ------
    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Wrapper function that executes the task function.

        Parameters:
        -----------
        cfg : DictConfig
            The configuration object.

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            A tuple containing metric and object dictionaries.
        """
        # execute the task
        try:
            dr = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            import wandb

            if wandb.run:
                log.info("Closing wandb!")
                wandb.finish()

            torch.cuda.empty_cache()
            gc.collect()

        return dr

    return wrap
