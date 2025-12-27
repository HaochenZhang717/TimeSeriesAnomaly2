#export CUDA_VISIBLE_DEVICES=4,5

# use this for full finetuning
#accelerate launch --config_file tutorials/finetune_demo/ds.yaml \
#    tutorials/finetune_demo/classification.py \
#    --base_path path to your ptbxl base folder \
#    --cache_dir path to cache directory for preprocessed dataset \
#    --mode full_finetuning \
#    --output_path path to store train log and checkpoint \

# #use this for linear_probing, svm, unsupervised_representation_learning
python classification.py \
    --mode "linear_probing" \
    --output_path "./" \
    --train_data_path "/root/tianyi/formal_experiment/mitdb/dsp_flow/impute_finetune_ckpt_lr1e-4/no_code_impute_samples.pth"\
    --test_data_path "../dataset_utils/ECG_Datasets/raw_data/mitdb106_test_data.pt"\
    --key_signal "all_samples" \
    --key_label "all_labels"

