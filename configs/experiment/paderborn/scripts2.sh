HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn/multilabel_exps/real_bearings=eval_bestparams_fft_4x1"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn/multilabel_exps/real_bearings=eval_bestparams_fft_2x3"
HYDRA_FULL_ERROR=1 python src/train.py -m "+experiment/paderborn/multilabel_exps/real_bearings=eval_bestparams_fft_1x4"
