
cd ..

DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_svdb/859.npz"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/V_test.jsonl"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/V_train.jsonl"]'
OUTDIR="../nn_eval/svdb_two_channels/real_data"

MAX_LEN_ANOMALY=360
LEN_WHOLE=800

echo "Running nn_eval: ${FIENTUNE_CKPT}"

python run_nn_evaluate_real_data.py \
    --seq_len ${LEN_WHOLE} \
    --feature_size 2 \
    --one_channel 0 \
    --feat_window_size 300 \
    --raw_data_paths ${DATA_PATHS} \
    --indices_paths_train ${FINETUNE_TRAIN_INDICES_PATHS} \
    --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
    --max_infill_length ${MAX_LEN_ANOMALY} \
    --out_dir "${OUTDIR}" \
    --gpu_id 0

