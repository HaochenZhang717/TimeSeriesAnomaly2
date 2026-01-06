


#CUDA_VISIBLE_DEVICES=0 bash timevae_all/cgats_mitdb.sh &
#CUDA_VISIBLE_DEVICES=1 bash timevae_all/cgats_qt.sh &
#CUDA_VISIBLE_DEVICES=2 bash timevae_all/cgats_svdb.sh &

#CUDA_VISIBLE_DEVICES=3 bash timevae_all/timevae_mitdb.sh &
#CUDA_VISIBLE_DEVICES=4 bash timevae_all/timevae_qt.sh &
#CUDA_VISIBLE_DEVICES=5 bash timevae_all/timevae_svdb.sh &


#CUDA_VISIBLE_DEVICES=3 bash diffusionts_baseline/diffts_mit106.sh &
#CUDA_VISIBLE_DEVICES=4 bash diffusionts_baseline/diffts_qt233.sh &
CUDA_VISIBLE_DEVICES=5 bash diffusionts_baseline/diffts_svdb.sh &


#CUDA_VISIBLE_DEVICES=0 bash flowts_baseline/flowts_mit.sh &
#CUDA_VISIBLE_DEVICES=1 bash flowts_baseline/flowts_qt233.sh &
CUDA_VISIBLE_DEVICES=2 bash flowts_baseline/flowts_svdb.sh &

bash dsp_our_method/dsp_flow_svdb_two_channels_mixed.sh &

CUDA_VISIBLE_DEVICES=3 dsp_no_code/dsp_flow_svdb_two_channels_no_code.sh &




