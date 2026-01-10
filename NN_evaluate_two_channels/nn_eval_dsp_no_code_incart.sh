
cd ..

DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_incart/I30.npz"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_incart/slide_windows_I30npz/V_test.jsonl"]'
FINETUNE_CKPT="../formal_experiment/incart_6_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4"

OUTDIR="../nn_eval/incart_6_channels/dspflow_no_code"

MAX_LEN_ANOMALY=600
LEN_WHOLE=800


python run_nn_evaluate.py \
    --seq_len ${LEN_WHOLE} \
    --feature_size 6 \
    --one_channel 0 \
    --feat_window_size 300 \
    --raw_data_paths ${DATA_PATHS} \
    --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
    --max_infill_length ${MAX_LEN_ANOMALY} \
    --ckpt_dir "${FINETUNE_CKPT}" \
    --out_dir "${OUTDIR}" \
    --generated_path "${FINETUNE_CKPT}/no_code_impute_samples.pth" \
    --gpu_id 0

cd ./NN_evaluate