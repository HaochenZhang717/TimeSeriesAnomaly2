
cd ..


DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_traffic/metro_traffic_data.npz"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_traffic/slide_windows_metro_traffic_datanpz/V_test.jsonl"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_traffic/slide_windows_metro_traffic_datanpz/V_train.jsonl"]'
OUTDIR="../nn_eval/traffic/real_data"



MAX_LEN_ANOMALY=24
LEN_WHOLE=72

echo "Running nn_eval: ${FIENTUNE_CKPT}"

python run_nn_evaluate_real_data.py \
    --seq_len ${LEN_WHOLE} \
    --feature_size 2 \
    --one_channel 0 \
    --feat_window_size 50 \
    --raw_data_paths ${DATA_PATHS} \
    --indices_paths_train ${FINETUNE_TRAIN_INDICES_PATHS} \
    --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
    --max_infill_length ${MAX_LEN_ANOMALY} \
    --out_dir "${OUTDIR}" \
    --gpu_id 0

