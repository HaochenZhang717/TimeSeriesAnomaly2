cd ..

## dsp_no_code
### MITDB No Code
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/mitdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/mitdb_two_channels/dsp_flow_no_code.jsonl"
#
### SVDB No Code
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/svdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/svdb_two_channels/dsp_flow_no_code.jsonl" \
#
### QTDB No Code
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/qtdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/qtdb_two_channels/dsp_flow_no_code.jsonl" \
#
### PV No Code
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/PV/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/pv/dsp_flow_no_code.jsonl" \
#
### Traffic No Code
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/traffic/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/traffic/dsp_flow_no_code.jsonl" \
#
#
#
## dsp_ours
## MITDB
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/mitdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/mitdb_two_channels/dsp_flow_mixed_K500.jsonl" \
#### SVDB
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/svdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/svdb_two_channels/dsp_flow_mixed_K500.jsonl" \
#### QTDB
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/qtdb_two_channels/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/qtdb_two_channels/dsp_flow_mixed_K500.jsonl" \
#
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/PV/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/PV/dsp_flow_mixed_K500.jsonl" \
#
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/traffic/dsp_flow_mixed_K500/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/traffic/dsp_flow_mixed_K500.jsonl" \


## timeVAE
### MITDB
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/mitdb_two_channels/TimeVAE/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/mitdb_two_channels/timevae.jsonl" \
### SVDB
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/svdb_two_channels/TimeVAE/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/svdb_two_channels/timevae.jsonl" \
#### QTDB
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/qtdb_two_channels/TimeVAE/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/qtdb_two_channels/timevae.jsonl" \
#
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/PV/TimeVAE/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/PV/timevae.jsonl" \
#
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/traffic/TimeVAE/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/traffic/timevae.jsonl" \

#### FlowTS
### MITDB
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/mitdb_two_channels/flowts/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/mitdb_two_channels/flowts.jsonl" \
### SVDB
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/svdb_two_channels/flowts/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/svdb_two_channels/flowts.jsonl" \
### QTDB
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/qtdb_two_channels/flowts/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/qtdb_two_channels/flowts.jsonl" \
#
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/PV/flowts/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/PV/flowts.jsonl" \
#
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/traffic/flowts/no_code_impute_from_scratch_ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/traffic/flowts.jsonl" \



## Diffusion-TS
### MITDB
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/mitdb_two_channels/diffusion_ts/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/mitdb_two_channels/diffts.jsonl" \
### SVDB
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/svdb_two_channels/diffusion_ts/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/svdb_two_channels/diffts.jsonl" \
### QTDB
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/qtdb_two_channels/diffusion_ts/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/qtdb_two_channels/diffts.jsonl" \
#
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/PV/diffusion_ts/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/PV/diffts.jsonl" \
#
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/traffic/diffusion_ts/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/traffic/diffts.jsonl" \
#
#
#
#
### C-GATS
#### MITDB
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/mitdb_two_channels/C-GATS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/mitdb_two_channels/cgats.jsonl" \
### SVDB
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/svdb_two_channels/C-GATS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/svdb_two_channels/cgats.jsonl" \
### QTDB
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/qtdb_two_channels/C-GATS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/qtdb_two_channels/cgats.jsonl" \
#
#
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/PV/C-GATS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/PV/cgats.jsonl" \
#
#
#python run_non_downstream_evaluate.py \
#--samples_path "../formal_experiment/traffic/C-GATS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/traffic/cgats.jsonl" \




### GenIAS
#### MITDB
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/mitdb_two_channels/GENIAS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/mitdb_two_channels/genias.jsonl" \
### SVDB
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/svdb_two_channels/GENIAS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/svdb_two_channels/genias.jsonl" \
### QTDB
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/qtdb_two_channels/GENIAS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/qtdb_two_channels/genias.jsonl" \
#
### PV
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/PV/GENIAS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/PV/genias.jsonl" \
#
## Traffic
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/traffic/GENIAS/ckpt_lr1e-4/no_code_impute_samples_non_downstream.pth" \
#--save_dir "../non_downstream_eval_result/traffic/genias.jsonl" \

## DSP_no_pretrain
#### MITDB
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/formal_experiment/dsp_no_pretrain_all_datasets/dsp_flow_mixed_K500_no_pretrain_mitdb/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream_test.pth" \
#--save_dir "../non_downstream_eval_result/mitdb_two_channels/dsp_no_pretrain.jsonl" \
### SVDB
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/formal_experiment/dsp_no_pretrain_all_datasets/dsp_flow_mixed_K500_no_pretrain_svdb/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream_test.pth" \
#--save_dir "../non_downstream_eval_result/svdb_two_channels/dsp_no_pretrain.jsonl" \
### QTDB
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/formal_experiment/dsp_no_pretrain_all_datasets/dsp_flow_mixed_K500_no_pretrain_qtdb/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream_test.pth" \
#--save_dir "../non_downstream_eval_result/qtdb_two_channels/dsp_no_pretrain.jsonl" \
#
### PV
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/formal_experiment/dsp_no_pretrain_all_datasets/dsp_flow_mixed_K500_no_pretrain_pv/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream_test.pth" \
#--save_dir "../non_downstream_eval_result/PV/dsp_no_pretrain.jsonl" \
#
## Traffic
#python run_non_downstream_evaluate.py \
#--samples_path "/Users/zhc/Documents/formal_experiment/dsp_no_pretrain_all_datasets/dsp_flow_mixed_K500_no_pretrain_traffic/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream_test.pth" \
#--save_dir "../non_downstream_eval_result/traffic/dsp_no_pretrain.jsonl" \


### MITDB
python run_non_downstream_evaluate.py \
--samples_path "/Users/zhc/Documents/formal_experiment/dsp_no_pretrain_all_datasets/dsp_flow_mixed_K500_no_pretrain_mitdb/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream_train.pth" \
--save_dir "../non_downstream_eval_result/mitdb_two_channels/dsp_no_pretrain.jsonl" \
## SVDB
python run_non_downstream_evaluate.py \
--samples_path "/Users/zhc/Documents/formal_experiment/dsp_no_pretrain_all_datasets/dsp_flow_mixed_K500_no_pretrain_svdb/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream_train.pth" \
--save_dir "../non_downstream_eval_result/svdb_two_channels/dsp_no_pretrain.jsonl" \
## QTDB
python run_non_downstream_evaluate.py \
--samples_path "/Users/zhc/Documents/formal_experiment/dsp_no_pretrain_all_datasets/dsp_flow_mixed_K500_no_pretrain_qtdb/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream_train.pth" \
--save_dir "../non_downstream_eval_result/qtdb_two_channels/dsp_no_pretrain.jsonl" \

## PV
python run_non_downstream_evaluate.py \
--samples_path "/Users/zhc/Documents/formal_experiment/dsp_no_pretrain_all_datasets/dsp_flow_mixed_K500_no_pretrain_pv/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream_train.pth" \
--save_dir "../non_downstream_eval_result/PV/dsp_no_pretrain.jsonl" \

# Traffic
python run_non_downstream_evaluate.py \
--samples_path "/Users/zhc/Documents/formal_experiment/dsp_no_pretrain_all_datasets/dsp_flow_mixed_K500_no_pretrain_traffic/impute_finetune_ckpt_lr1e-4/posterior_impute_samples_non_downstream_train.pth" \
--save_dir "../non_downstream_eval_result/traffic/dsp_no_pretrain.jsonl" \


cd non_downstream_eval