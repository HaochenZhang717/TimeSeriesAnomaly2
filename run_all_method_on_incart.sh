bash dsp_our_method/dsp_flow_incart_mixed.sh & # 0,1,2,3,4
CUDA_VISIBLE_DEVICES=5 bash dsp_no_code/dsp_flow_incart_no_code.sh &
CUDA_VISIBLE_DEVICES=6 bash flowts_baseline/flowts_incart.sh &
CUDA_VISIBLE_DEVICES=7 bash diffusionts_baseline/diffts_incart.sh &

CUDA_VISIBLE_DEVICES=6 bash timevae_all/cgats_incart.sh &
CUDA_VISIBLE_DEVICES=7 bash timevae_all/timevae_incart.sh &
