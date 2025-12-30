export CUDA_VISIBLE_DEVICES=5

cd ..

DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_svdb/859.npz"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/V_test.jsonl"]'
FINETUNE_CKPT="/root/tianyi/formal_experiment/svdb_one_channel/flowts/no_code_impute_from_scratch_ckpt_lr1e-4"
MAX_LEN_ANOMALY=360
LEN_WHOLE=800


python run_nn_evaluate.py \
    --seq_len ${LEN_WHOLE} \
    --feature_size 1 \
    --one_channel 1 \
    --raw_data_paths ${DATA_PATHS} \
    --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
    --max_infill_length ${MAX_LEN_ANOMALY} \
    --ckpt_dir "${FINETUNE_CKPT}" \
    --generated_path "${FINETUNE_CKPT}/no_code_impute_samples.pth" \
    --gpu_id 0

cd ./NN_evaluate