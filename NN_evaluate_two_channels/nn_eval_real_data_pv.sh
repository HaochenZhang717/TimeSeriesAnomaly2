
cd ..

DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_2.npz","./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_3.npz","./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_4.npz","./dataset_utils/ECG_datasets/raw_data_PV/2015_pv_sub_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2015_pv_sub_1.npz","./dataset_utils/ECG_datasets/raw_data_PV/2021_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2022_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2023_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2024_pv_live_0.npz","./dataset_utils/ECG_datasets/raw_data_PV/2025_pv_live_0.npz"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/V_train.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/V_test.jsonl"]'
OUTDIR="../nn_eval/PV/real_data"





MAX_LEN_ANOMALY=144
LEN_WHOLE=200

echo "Running nn_eval: ${FIENTUNE_CKPT}"

python run_nn_evaluate_real_data.py \
    --seq_len ${LEN_WHOLE} \
    --feature_size 1 \
    --one_channel 1 \
    --feat_window_size 100 \
    --raw_data_paths ${DATA_PATHS} \
    --indices_paths_train ${FINETUNE_TRAIN_INDICES_PATHS} \
    --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
    --max_infill_length ${MAX_LEN_ANOMALY} \
    --out_dir "${OUTDIR}" \
    --gpu_id 0

