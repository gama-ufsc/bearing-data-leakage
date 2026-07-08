# Bearing Data Leakage

This repository provides the complete experimental pipeline used in the paper

**J. P. Vieira et al.**  
*Towards a more realistic evaluation of machine learning models for
bearing fault diagnosis*  
Pre-print currently available on arXiv: https://arxiv.org/abs/2509.22267

The codebase enables reproducible data preprocessing, feature extraction, dataset splitting, model training, and evaluation with the methodology described in the paper. 

---

## Repository Overview

The project is organized around two main components:

- **Data pipeline**: responsible for preparing datasets before training (preprocessing, dataloaders, etc.).
- **Model pipeline**: responsible for training and evaluating shallow and deep learning models.

The pipelines are implemented using:
- **Hamilton** for function-based data and model pipelines.
- **Hydra** for configuration management and experiment definition.
- **Pytorch Lightining** for model training/evaluation.

Most of the experiments described in the paper can be reproduced using the configuration files provided in this repository.

---

## Regarding data splits

If you wish to simply reproduce experiments using the same validation/evaluation train-test splits from the paper, they are all available in `data/splits`. More details are given in `Important Notebooks` subsection.

## Environment Setup

This project uses **uv** for dependency management and virtual environment creation.

### Step 1 — Install `uv`

If `uv` is not installed, follow the official installation instructions:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

More details are available at:
https://docs.astral.sh/uv/

Verify installation:
```bash
uv --version
```

### Step 2 — Create the virtual environment

From the repository root directory, run:

```bash
uv sync
```

This command will:

- Create a virtual environment.

- Install all dependencies listed in pyproject.toml.

Alternatively, run: 

```bash
make install
```

Activate the environment if necessary:

```bash
source .venv/bin/activate
```

### Important Notebooks

The `notebooks/` directory contains the main preprocessing and analysis steps used in the study.

1. `process_datasets.ipynb` is **essential**: it loads the original raw dataset files (CWRU, Ottawa, and Paderborn); extracts and organizes metadata (bearing IDs, fault types, and operating conditions) and;  saves processed raw signals (`data/datasets/*`) in compressed formats (.bz2 / .pkl) and a metadata dataframe (`data/metadata/*`) for each dataset. 

Although CWRU is downloadable through this notebook, you will need to pre-download UORED-VAFCLS and Paderborn files (URL is provided in the notebook). **This step must be executed before any dataset splitting or model training.**

2. `create_splits_for_all_datasets.ipynb`: this notebook presents in detail the splitting methodology and generates train/test partitions. All tuning and evaluation splits are available in `data/splits`.

3. `feature_engineering.ipynb`: this notebook computes hand-crafted features for shallow machine learning models, including time-domain, frequency-domain, and envelope-based features. Feature dataframes are saved (`data/features/*`) in order to be used for training and evaluating models. 

4. **Shallow Model Notebooks**: application of our methodology into training/evaluating shallow learning models in the CWRU, PU and UORED-VAFCLS datasets.

5. `results.ipynb`: loads stored model outputs and logits (`data/saved_logits`) and computes ROC curves and Macro AUROC.


## Pipeline Structure

The project is structured as a configurable experiment pipeline using Hamilton and Hydra.

### Hamilton Pipelines

Hamilton is used to define pipelines as directed acyclic graphs (DAGs) of Python functions:

- **Data pipeline** (`src/data/data_pipeline.py`)

    Handles dataset loading, preprocessing, transformations, and feature extraction.

- **Model pipeline** (`src/model/model_pipeline.py`)

    Handles model definition, training, inference, and metric computation.


### Hydra Configuration System

Hydra manages configuration files located in the `configs/` directory.

Key configuration groups include:

- `dataset/` — dataset definitions and split functions

- `feature_pipeline/` — feature extraction configurations

- `modelmodule/` — model architectures, metrics and loss functions

- `trainer/` — Pytorch lightning training parameters

- `experiment/` — complete experiment definitions (e.g. evaluation in UORED-VAFCLS)

Each experiment is defined as a YAML file combining these components. Examples include:

```bash
configs/experiment/cwru/eval_fft.yaml
configs/experiment/ottawa/leakage_fft_noleakage.yaml
configs/experiment/paderborn/eval_envelope.yaml
```

These configuration files specify:

- Dataset

- Splitting strategy

- Input representation (e.g. time, FFT)

- Model

- Training parameters


## Running Experiments

After setting up the environment and processing the datasets, experiments can be executed using the Hydra-based training scripts.

Example:

```bash
python train.py -m "+experiment/cwru=eval_fft"
```

If unable to activate the environment, run:

```bash
./.venv/bin/python train.py -m "+experiment/cwru=eval_fft"
```


Hydra will:

- Load the selected configuration.

- Instantiate the data and model pipelines.

- Create multiruns (`-m`) that vary specific parameters defined in the experiment.

- Run training and evaluation for various runs. 

- Save results and logs locally and on **WANDB** (you must define your entity and project on `configs/train.yaml`). To run offline, without WANDB, set `run_offline: True` in train.yaml.


## Experiments in detail

The directory structure of experiments is organized by dataset as follows:


```bash
configs
└── experiment
├── cwru
│ ├── eval_fft.yaml      |
│ ├── eval_envelope.yaml | -> Training/evaluation experiments
│ ├── eval_time.yaml     |
│ ├── leakage_env_noleakage.yaml     |
│ ├── leakage_env_condition.yaml     | -> Data leakage experiments
│ ├── leakage_env_condition.yaml     |
│ └── ...                            |
│
├── ottawa
│ ├── diversity.yaml | -> Diversity experiments
│ ├── eval_fft.yaml         |
│ ├── eval_envelope.yaml    | -> Training/evaluation experiments
│ ├── eval_time.yaml        |
│ ├── leakage_fft_noleakage.yaml    |
│ ├── leakage_fft_bearing.yaml      | -> Data leakage experiments
│ ├── leakage_fft_segment.yaml      |
│ └── ...
│
└── paderborn
├── diversity.yaml
├── eval_fft.yaml
├── eval_envelope.yaml
├── eval_time.yaml
├── leakage_env_noleakage.yaml
└── ...
```

## Contact

For questions, issues, or suggestions related to this repository, please open an issue or contact the authors via e-mail (joaopaulovieirajpv@gmail.com, victor.bauler@gmail.com). 
