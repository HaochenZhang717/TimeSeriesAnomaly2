export CUDA_VISIBLE_DEVICES=7
model_sizes=("large" "base" "small")

for model_size in "${model_sizes[@]}"; do
  echo "Running classification with model size: ${model_size}"

  python classification.py \
    --mode "linear_probing" \
    --output_path "/root/tianyi/moment_eval/qtdb/dsp_flow_no_code/${model_size}" \
    --train_data_path "/root/tianyi/formal_experiment/qtdb_one_channel/dsp_flow/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples.pth" \
    --test_data_path "../dataset_utils/ECG_datasets/test_data/qtdb233_test_data.pt" \
    --key_signal "all_samples" \
    --key_label "all_labels" \
    --model_name "${model_size}" \
    --one_channel "true"

done