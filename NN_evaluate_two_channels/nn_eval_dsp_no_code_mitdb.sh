
cd ..

DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data/106.npz"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_test.jsonl"]'
FINETUNE_CKPT="/root/tianyi/formal_experiment/mitdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr1e-4"

OUTDIR="/root/tianyi/nn_eval/mitdb_two_channels/dspflow_no_code"

MAX_LEN_ANOMALY=800
LEN_WHOLE=1000


python run_nn_evaluate.py \
    --seq_len ${LEN_WHOLE} \
    --feature_size 2 \
    --one_channel 0 \
    --raw_data_paths ${DATA_PATHS} \
    --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
    --max_infill_length ${MAX_LEN_ANOMALY} \
    --ckpt_dir "${FINETUNE_CKPT}" \
    --out_dir "${OUTDIR}" \
    --generated_path "${FINETUNE_CKPT}/no_code_impute_samples.pth" \
    --gpu_id 0

cd ./NN_evaluate