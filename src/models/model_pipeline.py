from hamilton.function_modifiers import config
import lightning as L
from torch.utils.data import DataLoader
from typing import List, Union, Dict
import logging
import pandas as pd
from sklearn.base import BaseEstimator
import inspect


def _log_metrics(
    y_true: pd.Series,
    y_pred,
    y_proba,
    metrics: Dict,
    logger: logging.Logger,
    stage: str,
):
    scores = {}
    for metric_name, metric_func in metrics.items():
        logger.debug(f"Calculating {metric_name} for {stage}")

        metric_func_params = set(inspect.signature(metric_func).parameters.keys())

        if "y_pred" in metric_func_params:
            metric = metric_func(y_true, y_pred)
        elif "y_proba" in metric_func_params:
            metric = metric_func(y_true, y_proba)
        else:
            raise ValueError(
                "Metric function must have either `y_pred` or `y_proba` as a parameter"
            )

        logger.info(f"{stage} {metric_name}: {metric}")

        scores[metric_name] = metric

    return scores


@config.when(pipeline="traditional")
def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    logger: logging.Logger,
    evaluation_metrics: Dict,
) -> BaseEstimator:
    logger.info("Starting training")
    model.fit(X_train, y_train)
    logger.info("Training finished")

    logger.debug("Calculating metrics for train")

    train_pred = model.predict(X_train)

    if hasattr(model, "predict_proba"):
        train_proba = model.predict_proba(X_train)[:, 1]
    elif hasattr(model, "decision_function"):
        train_proba = model.decision_function(X_train)
    else:
        raise ValueError(
            "Model does not have a predict_proba or decision_function method"
        )

    _ = _log_metrics(
        y_train, train_pred, train_proba, evaluation_metrics, logger, "train"
    )

    return model


@config.when(pipeline="traditional")
def inference(
    train_model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    logger: logging.Logger,
    evaluation_metrics: Dict,
) -> pd.Series:
    logger.info("Starting inference")

    logger.debug("Calculating metrics for test")
    test_pred = train_model.predict(X_test)

    if hasattr(train_model, "predict_proba"):
        test_proba = train_model.predict_proba(X_test)[:, 1]
    elif hasattr(train_model, "decision_function"):
        test_proba = train_model.decision_function(X_test)
    else:
        raise ValueError(
            "Model does not have a predict_proba or decision_function method"
        )

    scores = _log_metrics(
        y_test, test_pred, test_proba, evaluation_metrics, logger, "test"
    )

    return {"scores": scores}


@config.when(pipeline="deep_learning")
def trainer_fit(
    modelmodule: L.LightningModule,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    trainer: L.Trainer,
    load_model: bool = False,
) -> List[Union[L.Trainer, L.LightningModule]]:
    if load_model:
        return trainer, modelmodule
    else:  # train from scratch
        if validation_dataloader is not None:
            trainer.fit(modelmodule, train_dataloader, validation_dataloader)
        else:
            trainer.fit(modelmodule, train_dataloader)
    
    return trainer, modelmodule


@config.when(pipeline="deep_learning")
def trainer_test(
    trainer_fit: List[Union[L.Trainer, L.LightningModule]],
    test_dataloader: DataLoader,
    load_model: bool = False,
    model_path: str = None,
) -> L.Trainer:
    trainer, modelmodule = trainer_fit
    if load_model:
        if model_path is None:
            raise ValueError("model_path must be provided when load_model is True")
        trainer.test(modelmodule, test_dataloader, ckpt_path=model_path)
    else:
        trainer.test(modelmodule, test_dataloader)
    return trainer
