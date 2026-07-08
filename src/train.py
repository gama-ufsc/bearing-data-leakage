"""
Script for training and evaluating a model using Hamilton, PyTorch Lightning and Hydra.
"""

import logging
from typing import Dict, Optional
import os

import autorootcwd  # noqa
import hydra
import lightning as L
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from hamilton import driver
from src.data import data_pipeline
from src.models import model_pipeline

torch.set_float32_matmul_precision("medium")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import hydra_decorators  # noqa: E402

logging.basicConfig(
    filename="logs/train.log",
    filemode="a",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


log = logging.getLogger(__name__)

os.environ["WANDB_IGNORE_GLOBS"] = (
    "wandb-metadata.json,requirements.txt,output.log,wandb-summary.json,config.yaml"
)


@hydra_decorators.task_wrapper
def train(cfg: DictConfig) -> Dict[str, float]:
    """
    Train and evaluate a model using the provided configuration with hydra.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra using yaml files.

    Returns
    -------
    Dict[str, float]
        Dictionary with the metrics obtained during training and testing.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    pipeline = cfg_dict.get("pipeline")
    cfg_dict["logger"] = log

    log.info("Instantiating datasets")
    cfg_dict["datasets"] = hydra.utils.instantiate(cfg.datasets, _convert_="partial")

    if pipeline == "deep_learning":
        log.info(f"Instantiating modelmodule <{cfg.modelmodule._target_}>")
        cfg_dict["modelmodule"] = hydra.utils.instantiate(
            cfg.modelmodule, _convert_="partial"
        )

        log.info("Instantiating transform")
        cfg_dict["transform"] = hydra.utils.instantiate(cfg.transform)

        log.info("Instantiating normalization function")
        cfg_dict["normalization_function"] = hydra.utils.instantiate(
            cfg.normalization_function
        )

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        cfg_dict["trainer"] = hydra.utils.instantiate(cfg.trainer)

        if cfg.collate is not None:
            try:
                log.info(f"Instantiating collate <{cfg.collate._target_}>")
                cfg_dict["collate_train"] = hydra.utils.instantiate(cfg.collate.train)
                cfg_dict["collate_val"] = hydra.utils.instantiate(cfg.collate.val)
            except Exception:
                print("Failed to instatiante collate function")
                cfg_dict["collate_train"] = None
                cfg_dict["collate_val"] = None

        dr = (
            driver.Builder()
            .with_config(
                {
                    "pipeline": pipeline,
                }
            )
            .with_modules(data_pipeline, model_pipeline)
            .build()
        )

        cfg_dict.pop("pipeline")
        dr.execute(["trainer_test"], inputs=cfg_dict)
    elif pipeline == "traditional":
        pass
    else:
        raise ValueError(
            f"Invalid pipeline: `{pipeline}`, must be 'deep_learning' or 'traditional'"
        )

    return dr


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main function to be called by Hydra.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra using yaml files.

    Returns
    -------
    Optional[float]
        The value of the optimized metric obtained during training
    """
    dr = train(cfg)

    return dr


if __name__ == "__main__":
    main()
