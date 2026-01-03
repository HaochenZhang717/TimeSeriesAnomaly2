



cd ..
# MITDB
CUDA_VISIBLE_DEVICES=0 \
python run_non_downstream_evaluate.py \
--fake_data_path \
--real_data_path "/root/tianyi/TimeSeriesAnomaly2/dataset_utils/ECG_datasets/test_data/mitdb106_train_data.pt" \
--num_iters 5 \
--save_dir "/root/tianyi/non_downstream_eval_result/mitdb/dsp_flow.jsonl" \




cd non_downstream_eval