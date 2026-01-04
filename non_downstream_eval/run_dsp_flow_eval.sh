cd ..

# MITDB No Code
CUDA_VISIBLE_DEVICES=6 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/mitdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/mitdb_two_channels/dsp_flow_no_code.jsonl"
# QTDB No Code
CUDA_VISIBLE_DEVICES=6 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/qtdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/qtdb_two_channels/dsp_flow_no_code.jsonl" \
## SVDB No Code
CUDA_VISIBLE_DEVICES=6 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/svdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/qtdb_two_channels/dsp_flow_no_code.jsonl" \


## MITDB
CUDA_VISIBLE_DEVICES=7 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/mitdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/mitdb_two_channels/dsp_flow_mixed_K500.jsonl" \
## QTDB
CUDA_VISIBLE_DEVICES=7 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/qtdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/qtdb_two_channels/dsp_flow_mixed_K500.jsonl" \
## SVDB
CUDA_VISIBLE_DEVICES=7 \
python run_non_downstream_evaluate.py \
--samples_path "/root/tianyi/formal_experiment/svdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth" \
--save_dir "/root/tianyi/non_downstream_eval_result/qtdb_two_channels/dsp_flow_mixed_K500.jsonl" \




cd non_downstream_eval