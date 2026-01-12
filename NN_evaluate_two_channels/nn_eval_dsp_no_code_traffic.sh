
cd ..

DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_traffic/metro_traffic_data.npz"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_traffic/slide_windows_metro_traffic_datanpz/V_test.jsonl"]'
FINETUNE_CKPT="../formal_experiment/traffic/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr${LR}"

OUTDIR="../nn_eval/traffic/dspflow_no_code"

LEN_WHOLE=72
MAX_LEN_ANOMALY=48

python run_nn_evaluate_new.py \
    --seq_len ${LEN_WHOLE} \
    --feature_size 1 \
    --one_channel 1 \
    --feat_window_size 50 \
    --raw_data_paths ${DATA_PATHS} \
    --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
    --max_infill_length ${MAX_LEN_ANOMALY} \
    --ckpt_dir "${FINETUNE_CKPT}" \
    --out_dir "${OUTDIR}" \
    --generated_path "${FINETUNE_CKPT}/principle_no_code_impute_samples.pth" \
    --gpu_id 0


cd ./NN_evaluate