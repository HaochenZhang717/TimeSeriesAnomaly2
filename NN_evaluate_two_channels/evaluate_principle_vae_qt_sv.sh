



CUDA_VISIBLE_DEVICES=2 bash nn_eval_timevae_qtdb.sh &
CUDA_VISIBLE_DEVICES=3 bash nn_eval_timevae_svdb.sh &

CUDA_VISIBLE_DEVICES=0 bash nn_eval_cgats_qtdb.sh &
CUDA_VISIBLE_DEVICES=1 bash nn_eval_cgats_svdb.sh &