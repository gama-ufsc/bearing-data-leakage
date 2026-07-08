HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=leakage_time_noleakage"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=leakage_time_bearing"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=leakage_time_segment"

HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=leakage_fft_noleakage"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=leakage_fft_bearing"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=leakage_fft_segment"
 
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=leakage_env_noleakage"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=leakage_env_bearing"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=leakage_env_segment"