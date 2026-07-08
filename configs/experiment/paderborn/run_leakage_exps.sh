HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn=leakage_time_noleakage"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn=leakage_time_condition"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn=leakage_time_repetition"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn=leakage_time_segmentation"

HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn=leakage_fft_noleakage"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn=leakage_fft_condition"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn=leakage_fft_repetition"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn=leakage_fft_segmentation"

HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn=leakage_env_noleakage"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn=leakage_env_condition"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn=leakage_env_repetition"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn=leakage_env_segmentation"
