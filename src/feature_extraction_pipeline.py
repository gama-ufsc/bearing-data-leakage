# Código para feature engineering usando hydra
import autorootcwd  # noqa
import logging
import hydra
from hamilton import driver

from src.data.feature_engineering import feature_extraction_pipeline
import rootutils
from omegaconf import DictConfig, OmegaConf
from sklearn import set_config

set_config(transform_output="pandas")
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(
    filename="logs/feature_engineering.log",
    filemode="a",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


logger = logging.getLogger("Feature Engineering")


@hydra.main(
    config_path="../configs",
    config_name="feature_engineering",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    logger.info("Starting feature engineering pipeline...")

    logger.info("Instantiating hydra config...")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    feature_pipeline = hydra.utils.instantiate(
        cfg.feature_pipeline, _convert_="partial"
    )
    update_feature_store = hydra.utils.instantiate(cfg.update_feature_store)
    cfg_dict["datasets"] = hydra.utils.instantiate(cfg.datasets, _convert_="partial")

    cfg_dict.update(
        dict(
            feature_pipeline=feature_pipeline,
            update_feature_store=update_feature_store,
            logger=logger,
        )
    )

    dr = (
        driver.Builder()
        .with_config({})
        .with_modules(feature_extraction_pipeline)
        .build()
    )

    dr.execute(final_vars=["save_on_feature_store"], inputs=cfg_dict)


if __name__ == "__main__":
    main()
