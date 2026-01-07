
#CUDA_VISIBLE_DEVICES=0 bash nn_eval_dsp_no_code_mitdb.sh &
#CUDA_VISIBLE_DEVICES=1 bash nn_eval_dsp_no_code_qtdb.sh &
#CUDA_VISIBLE_DEVICES=2 bash nn_eval_dsp_no_code_svdb.sh &
#
#CUDA_VISIBLE_DEVICES=3 bash nn_eval_dsp_mitdb.sh 500 &
#CUDA_VISIBLE_DEVICES=4 bash nn_eval_dsp_qtdb.sh 500 &
#CUDA_VISIBLE_DEVICES=5 bash nn_eval_dsp_svdb.sh 500 &


CUDA_VISIBLE_DEVICES=6 bash nn_eval_diffts_mitdb.sh &
CUDA_VISIBLE_DEVICES=7 bash nn_eval_diffts_qtdb.sh &
CUDA_VISIBLE_DEVICES=0 bash nn_eval_diffts_svdb.sh &

CUDA_VISIBLE_DEVICES=1 bash nn_eval_flowts_mitdb.sh &
CUDA_VISIBLE_DEVICES=2 bash nn_eval_flowts_qtdb.sh &
CUDA_VISIBLE_DEVICES=3 bash nn_eval_flowts_svdb.sh &

CUDA_VISIBLE_DEVICES=4 bash nn_eval_timevae_mitdb.sh &
CUDA_VISIBLE_DEVICES=5 bash nn_eval_timevae_qtdb.sh &
CUDA_VISIBLE_DEVICES=6 bash nn_eval_timevae_svdb.sh &

CUDA_VISIBLE_DEVICES=7 bash nn_eval_cgats_mitdb.sh &
CUDA_VISIBLE_DEVICES=0 bash nn_eval_cgats_qtdb.sh &
CUDA_VISIBLE_DEVICES=1 bash nn_eval_cgats_svdb.sh &