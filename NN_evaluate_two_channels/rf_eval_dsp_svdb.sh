#!/bin/bash
set -e

# =========================
# Read arguments
# =========================
NUM_CODES=$1

cd ..

DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_svdb/859.npz"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/V_test.jsonl"]'
FINETUNE_CKPT="/root/tianyi/formal_experiment/svdb_two_channels/dsp_flow_mixed_K${NUM_CODES}/impute_finetune_ckpt_lr1e-4"
OUTDIR="/root/tianyi/nn_eval/svdb_two_channels/dsp_flow_mixed_K${NUM_CODES}"

MAX_LEN_ANOMALY=360
LEN_WHOLE=800


python run_rf_evaluate.py \
    --seq_len ${LEN_WHOLE} \
    --feature_size 2 \
    --one_channel 0 \
    --feat_window_size 300 \
    --raw_data_paths ${DATA_PATHS} \
    --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
    --max_infill_length ${MAX_LEN_ANOMALY} \
    --ckpt_dir "${FINETUNE_CKPT}" \
    --out_dir "${OUTDIR}" \
    --generated_path "${FINETUNE_CKPT}/principle_samples.pth" \
    --gpu_id 0

cd ./NN_evaluate