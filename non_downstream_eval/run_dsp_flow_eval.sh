cd ..

## dsp_no_code
## MITDB No Code
#CUDA_VISIBLE_DEVICES=6 \
#python run_non_downstream_evaluate.py \
#--samples_path "/root/tianyi/formal_experiment/mitdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "/root/tianyi/non_downstream_eval_result/mitdb_two_channels/dsp_flow_no_code.jsonl"
## QTDB No Code
#CUDA_VISIBLE_DEVICES=6 \
#python run_non_downstream_evaluate.py \
#--samples_path "/root/tianyi/formal_experiment/qtdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "/root/tianyi/non_downstream_eval_result/qtdb_two_channels/dsp_flow_no_code.jsonl" \
### SVDB No Code
#CUDA_VISIBLE_DEVICES=6 \
#python run_non_downstream_evaluate.py \
#--samples_path "/root/tianyi/formal_experiment/svdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "/root/tianyi/non_downstream_eval_result/svdb_two_channels/dsp_flow_no_code.jsonl" \

## dsp_ours
### MITDB
#CUDA_VISIBLE_DEVICES=7 \
#python run_non_downstream_evaluate.py \
#--samples_path "/root/tianyi/formal_experiment/mitdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth" \
#--save_dir "/root/tianyi/non_downstream_eval_result/mitdb_two_channels/dsp_flow_mixed_K500.jsonl" \
### QTDB
#CUDA_VISIBLE_DEVICES=7 \
#python run_non_downstream_evaluate.py \
#--samples_path "/root/tianyi/formal_experiment/qtdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth" \
#--save_dir "/root/tianyi/non_downstream_eval_result/qtdb_two_channels/dsp_flow_mixed_K500.jsonl" \
### SVDB
#CUDA_VISIBLE_DEVICES=7 \
#python run_non_downstream_evaluate.py \
#--samples_path "/root/tianyi/formal_experiment/svdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth" \
#--save_dir "/root/tianyi/non_downstream_eval_result/svdb_two_channels/dsp_flow_mixed_K500.jsonl" \



# timeVAE
## MITDB
CUDA_VISIBLE_DEVICES=7 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/mitdb_two_channels/TimeVAE/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/mitdb_two_channels/timevae.jsonl" \
## QTDB
CUDA_VISIBLE_DEVICES=7 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/qtdb_two_channels/TimeVAE/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/qtdb_two_channels/timevae.jsonl" \
## SVDB
CUDA_VISIBLE_DEVICES=7 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/svdb_two_channels/TimeVAE/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/svdb_two_channels/timevae.jsonl" \


## FlowTS
### MITDB
#CUDA_VISIBLE_DEVICES=7 \
#python run_non_downstream_evaluate.py \
#--samples_path "/root/tianyi/formal_experiment/mitdb_two_channels/flowts/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "/root/tianyi/non_downstream_eval_result/mitdb_two_channels/flowts.jsonl" \
### QTDB
#CUDA_VISIBLE_DEVICES=7 \
#python run_non_downstream_evaluate.py \
#--samples_path "/root/tianyi/formal_experiment/qtdb_two_channels/flowts/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "/root/tianyi/non_downstream_eval_result/qtdb_two_channels/flowts.jsonl" \
### SVDB
#CUDA_VISIBLE_DEVICES=7 \
#python run_non_downstream_evaluate.py \
#--samples_path "/root/tianyi/formal_experiment/svdb_two_channels/flowts/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "/root/tianyi/non_downstream_eval_result/svdb_two_channels/flowts.jsonl" \

# Diffusion-TS
## MITDB
CUDA_VISIBLE_DEVICES=7 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/mitdb_two_channels/diffusion_ts/ckpt_lr1e-4/no_code_impute_samples.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/mitdb_two_channels/diffts.jsonl" \
## QTDB
CUDA_VISIBLE_DEVICES=7 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/qtdb_two_channels/diffusion_ts/ckpt_lr1e-4/no_code_impute_samples.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/qtdb_two_channels/diffts.jsonl" \
## SVDB
CUDA_VISIBLE_DEVICES=7 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/svdb_two_channels/diffusion_ts/ckpt_lr1e-4/no_code_impute_samples.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/svdb_two_channels/diffts.jsonl" \


# C-GATS
## MITDB
CUDA_VISIBLE_DEVICES=7 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/mitdb_two_channels/cgats/finetune_ckpt/no_code_impute_samples_non_downstream.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/mitdb_two_channels/cgats.jsonl" \
## QTDB
CUDA_VISIBLE_DEVICES=7 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/qtdb_two_channels/cgats/finetune_ckpt/no_code_impute_samples_non_downstream.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/qtdb_two_channels/cgats.jsonl" \
## SVDB
CUDA_VISIBLE_DEVICES=7 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/svdb_two_channels/cgats/finetune_ckpt/no_code_impute_samples_non_downstream.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/svdb_two_channels/cgats.jsonl" \



cd non_downstream_eval