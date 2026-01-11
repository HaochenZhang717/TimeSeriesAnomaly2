
cd ..

DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_qtdb/sel233.npz"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_test.jsonl"]'
FINETUNE_CKPT="../formal_experiment/qtdb_two_channels/C-GATS/ckpt_lr1e-4"
OUTDIR="../nn_eval/qtdb_two_channels/C-GATS"

MAX_LEN_ANOMALY=450
LEN_WHOLE=600


python run_nn_evaluate.py \
    --seq_len ${LEN_WHOLE} \
    --feature_size 2 \
    --one_channel 0 \
    --feat_window_size 300 \
    --raw_data_paths ${DATA_PATHS} \
    --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
    --max_infill_length ${MAX_LEN_ANOMALY} \
    --ckpt_dir "${FINETUNE_CKPT}" \
    --out_dir "${OUTDIR}" \
    --generated_path "${FINETUNE_CKPT}/principle_no_code_impute_samples.pth" \
    --gpu_id 0

cd ./NN_evaluate