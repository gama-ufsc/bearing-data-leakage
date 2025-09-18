#HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=baseline"
#HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=rg_multiplier_cdcn"
#HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=rg_multiplier_DCASE2018"

#HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=rg_multiplier_wdtcnn"
#HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=rg_multiplier_resnet1d"
#HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=rg"
#HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=rcrg"

#HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=eval_fft_bestsetup"
#HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=eval_time_bestsetup"



#HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=leakage_time_0"
#HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=leakage_time_1"
#HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/ottawa=leakage_time_segment"

./.venv/bin/python src/train.py -m "+experiment/ottawa=diversity1x4"
