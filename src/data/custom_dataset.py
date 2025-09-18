from typing import Callable, List, Literal
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from hydra.utils import instantiate


class VibrationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        stage: str,
        transform: Callable,
        channels_output: int = 3,
        segment_length: int = 11500,
        overlap_pct: float = 0.97,
        label_names: List[str] = ["label"],  # noqa: B006
        signal_name: str = "signal",
        segmentation_strategy: Literal[
            "full_signal", "fixed_segments", "overlap"
        ] = "overlap",
        dataset_multiplier: int = 1,
        preprocess_function: partial = None,
        augs_pre_repr_transform: Callable = None,
        augs_post_repr_transform: Callable = None,
    ):
        self.channels_output = channels_output
        self.transform = transform
        self.segment_length = segment_length
        self.overlap = int(overlap_pct * segment_length)
        self.stride = segment_length - self.overlap
        self.label_names = label_names
        self.signal_name = signal_name
        self.preprocess_function = preprocess_function
        self.segmentation_strategy = segmentation_strategy
        self.dataset_multiplier = dataset_multiplier
        self.augs_pre_repr_transform = (
            instantiate(augs_pre_repr_transform)
            if augs_pre_repr_transform is not None
            else None
        )
        self.augs_post_repr_transform = (
            instantiate(augs_post_repr_transform)
            if augs_post_repr_transform is not None
            else None
        )

        self.stage = stage
        if self.preprocess_function is not None:
            print("Preprocessing data...")
            df = self.preprocess_function(df, stage=self.stage)
        else:
            print("No preprocessing function was provided.")
        self.signals, self.segments = self._get_segments(df, segmentation_strategy)

        # Check dataset multiplier
        if self.dataset_multiplier < 1:
            raise ValueError(
                f"Dataset multiplier must be greater than or equal to 1, but got {self.dataset_multiplier}."
            )
        elif not isinstance(self.dataset_multiplier, int):
            raise ValueError(
                f"Dataset multiplier must be an integer, but got type: {type(self.dataset_multiplier)}."
            )

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # (signal idx, segment idx, label)
        (i, j, label) = self.segments[idx]
        preprocessed_items = []

        for _, signal in enumerate(self.signals[i]):
            if self.segmentation_strategy == "full_signal":
                segment = signal
            else:
                segment = signal[
                    j * self.stride : j * self.stride + self.segment_length
                ]

            segment = torch.FloatTensor(segment)
            if self.augs_pre_repr_transform is not None:
                segment = self.augs_pre_repr_transform(segment)
            transformed = self.transform(segment)
            if self.augs_post_repr_transform is not None:
                transformed = self.augs_post_repr_transform(transformed)
            preprocessed_items.append(transformed)

        # preprocessed_items = np.array(preprocessed_items)
        if isinstance(preprocessed_items[0], np.ndarray):
            preprocessed_items = tuple(
                torch.from_numpy(arr) for arr in preprocessed_items
            )
        preprocessed_items = torch.stack(preprocessed_items)

        shape = preprocessed_items.shape

        if self.channels_output == 3:
            if shape[0] == 1:
                preprocessed_items = np.repeat(
                    preprocessed_items, axis=0, repeats=3
                )  # REPEATING CHANNEL 3 TIMES (RGB)

        elif self.channels_output == 1:
            if shape[0] > 1:
                raise ValueError(
                    f"Channels output is 1, but the processed signal has {shape[0]} channels."
                )
        else:
            raise ValueError("Only 1 or 3 channels are supported for now.")

        return {
            "X": preprocessed_items,
            "label": torch.FloatTensor(label),
            "idx": idx,  # Adicionando idx para facilitar a identificação do segmento pra debug
        }

    def _get_segments(self, df, segmentation_strategy):
        signals = []
        dataset_segments = []

        df = df.copy()
        # Remove casos com labels que não fazem sentido (casos multiclasse)
        df = df.dropna(subset=self.label_names)
        labels = df[self.label_names].values

        for i in range(len(df)):
            # Se self.signal_name for uma lista, extrai os sinais de cada coluna
            if isinstance(self.signal_name, list):
                signal = [df.iloc[i][nome] for nome in self.signal_name]
            else:
                signal = df.iloc[i][self.signal_name]

            label = labels[i]

            duration = df.iloc[i]["duration"]
            sample_rate = df.iloc[i]["fs"]
            total_samples = int(duration * int(sample_rate))

            if len(signal) < total_samples:
                # Preenche com zeros se o sinal for menor que o esperado
                # print(total_samples - len(signal))
                signal = np.append(signal, [0] * (total_samples - len(signal)))

            if segmentation_strategy == "fixed_segments":
                # Para cada sinal (ou cada canal, no caso de múltiplos sinais), aplica crop/zero-padding
                if isinstance(signal, list):
                    processed_signal = []
                    for s in signal:
                        if len(s) > self.segment_length:
                            proc = s[: self.segment_length]
                        elif len(s) < self.segment_length:
                            proc = np.pad(
                                s, (0, self.segment_length - len(s)), mode="constant"
                            )
                        else:
                            proc = s
                        processed_signal.append(proc)
                else:
                    if len(signal) > self.segment_length:
                        processed_signal = signal[: self.segment_length]
                    elif len(signal) < self.segment_length:
                        processed_signal = np.pad(
                            signal,
                            (0, self.segment_length - len(signal)),
                            mode="constant",
                        )
                    else:
                        processed_signal = signal

                # Mantém a estrutura original de signals
                if isinstance(signal, list):
                    signals += [processed_signal]
                else:
                    signals += [[processed_signal]]

                dataset_segments += [(i, 0, label)]  # Apenas um segmento por sinal

            elif segmentation_strategy == "overlap":
                # Modo original: segmentação com overlap
                if isinstance(signal, list):
                    signals += [signal]
                else:
                    signals += [[signal]]
                # Supondo que todos os canais tenham o mesmo tamanho, usamos o primeiro para calcular os segmentos
                if isinstance(signal, list):
                    length = len(signal[0])
                else:
                    length = len(signal)
                num_segments = 1 + (length - self.segment_length) // (
                    self.segment_length - self.overlap
                )
                num_segments = max(1, num_segments)  # Garante pelo menos um segmento
                for j in range(num_segments):
                    dataset_segments += [(i, j, label)]
            elif segmentation_strategy == "full_signal":
                # Para cada sinal (ou cada canal, no caso de múltiplos sinais), extrai o sinal completo
                if isinstance(signal, list):
                    processed_signal = []
                    for s in signal:
                        proc = s
                        processed_signal.append(proc)
                else:
                    processed_signal = signal

                # Mantém a estrutura original de signals
                if isinstance(signal, list):
                    signals += [processed_signal]
                else:
                    signals += [[processed_signal]]

                dataset_segments += [
                    (i, 0, label)
                ]  # Apenas um segmento por sinal (sinal completo)
            else:
                raise ValueError(
                    f"Segmentation strategy {segmentation_strategy} not supported."
                )

        print(
            f"Dataset created with {len(signals)} signals and {len(dataset_segments)} segments."
        )
        if self.dataset_multiplier == 1:
            return signals, dataset_segments
        else:
            new_dataset_segments = []
            for _ in range(self.dataset_multiplier):
                new_dataset_segments += dataset_segments

            dataset_segments = new_dataset_segments

            return signals, dataset_segments
