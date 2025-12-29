#export CUDA_VISIBLE_DEVICES=4,5

# use this for full finetuning
#accelerate launch --config_file tutorials/finetune_demo/ds.yaml \
#    tutorials/finetune_demo/classification.py \
#    --base_path path to your ptbxl base folder \
#    --cache_dir path to cache directory for preprocessed dataset \
#    --mode full_finetuning \
#    --output_path path to store train log and checkpoint \

# #use this for linear_probing, svm, unsupervised_representation_learning
#python classification.py \
#    --mode "linear_probing" \
#    --output_path "./dsp_flow" \
#    --train_data_path "/root/tianyi/formal_experiment/qtdb/dsp_flow/impute_finetune_ckpt_lr1e-4/posterior_impute_samples.pth"\
#    --test_data_path "../dataset_utils/ECG_datasets/test_data/qtdb233_test_data.pt"\
#    --key_signal "all_samples" \
#    --key_label "all_labels"


model_sizes=("large" "base" "small")

for model_size in "${model_sizes[@]}"; do
  echo "Running classification with model size: ${model_size}"

  python classification.py \
      --mode "linear_probing" \
      --output_path "./qtdb/dsp_flow/${model_size}" \
      --train_data_path "/root/tianyi/formal_experiment/qtdb_one_channel/dsp_flow/impute_finetune_ckpt_lr1e-4/posterior_impute_samples.pth"\
      --test_data_path "../dataset_utils/ECG_datasets/test_data/qtdb233_test_data.pt"\
      --key_signal "all_samples" \
      --key_label "all_labels" \
      --model_name ${model_size} \
      --one_channel "true"

  python classification.py \
    --mode "linear_probing" \
    --output_path "./qtdb/dsp_flow_no_code/${model_size}" \
    --train_data_path "/root/tianyi/formal_experiment/qtdb_one_channel/dsp_flow/no_code_impute_finetune_ckpt_lr1e-4/no_code_impute_samples.pth" \
    --test_data_path "../dataset_utils/ECG_datasets/test_data/qtdb233_test_data.pt" \
    --key_signal "all_samples" \
    --key_label "all_labels" \
    --model_name "${model_size}" \
    --one_channel "true"

done