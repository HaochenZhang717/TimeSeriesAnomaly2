#CUDA_VISIBLE_DEVICES=0 bash nn_eval_dsp_mitdb.sh 200 &
#CUDA_VISIBLE_DEVICES=1 bash nn_eval_dsp_mitdb.sh 300 &
#CUDA_VISIBLE_DEVICES=2 bash nn_eval_dsp_mitdb.sh 400 &
#CUDA_VISIBLE_DEVICES=3 bash nn_eval_dsp_mitdb.sh 500 &

#CUDA_VISIBLE_DEVICES=0 bash nn_eval_dsp_qtdb.sh 200 &
#CUDA_VISIBLE_DEVICES=1 bash nn_eval_dsp_qtdb.sh 300 &
#CUDA_VISIBLE_DEVICES=2 bash nn_eval_dsp_qtdb.sh 400 &
#CUDA_VISIBLE_DEVICES=3 bash nn_eval_dsp_qtdb.sh 500 &

#CUDA_VISIBLE_DEVICES=0 bash nn_eval_dsp_svdb.sh 200 &
#CUDA_VISIBLE_DEVICES=1 bash nn_eval_dsp_svdb.sh 300 &
#CUDA_VISIBLE_DEVICES=2 bash nn_eval_dsp_svdb.sh 400 &
#CUDA_VISIBLE_DEVICES=3 bash nn_eval_dsp_svdb.sh 500 &

#CUDA_VISIBLE_DEVICES=4 bash nn_eval_dsp_no_code_svdb.sh &
#CUDA_VISIBLE_DEVICES=5 bash nn_eval_dsp_no_code_mitdb.sh &
#CUDA_VISIBLE_DEVICES=6 bash nn_eval_dsp_no_code_qtdb.sh &

CUDA_VISIBLE_DEVICES=0 bash nn_eval_dsp_mitdb.sh 500 &
CUDA_VISIBLE_DEVICES=1 bash nn_eval_dsp_qtdb.sh 500 &
CUDA_VISIBLE_DEVICES=2 bash nn_eval_dsp_svdb.sh 500 &
CUDA_VISIBLE_DEVICES=3 bash nn_eval_dsp_no_code_svdb.sh &
CUDA_VISIBLE_DEVICES=4 bash nn_eval_dsp_no_code_mitdb.sh &
CUDA_VISIBLE_DEVICES=5 bash nn_eval_dsp_no_code_qtdb.sh &
