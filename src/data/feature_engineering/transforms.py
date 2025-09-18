from typing import Dict, List

from feature_extraction import (
    MetricConfig,
    SignalRepresentationConfig,
    Transform,
    TransformConfig,
)


def create_transform(metrics: List[Dict], representation: Dict):
    metrics_config = [MetricConfig(**metric) for metric in metrics]

    signal_representation_config = SignalRepresentationConfig(**representation)

    transform_config = TransformConfig(
        signal_representation=signal_representation_config,
        metrics=metrics_config,
    )

    transformation = Transform(transform_config)

    return transformation
