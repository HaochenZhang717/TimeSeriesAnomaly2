export CUDA_VISIBLE_DEVICES=5

cd ..

DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data/106.npz"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_test.jsonl"]'
FINETUNE_CKPT="/root/tianyi/formal_experiment/mitdb_one_channel/diffusion_ts/ckpt_lr1e-4"
OUTDIR="/root/tianyi/ml_eval/mitdb/diffusion_ts"
MAX_LEN_ANOMALY=800
LEN_WHOLE=1000
FEAT_WINDOW_SIZE=100

python run_ml_evaluate.py \
    --seq_len ${LEN_WHOLE} \
    --feat_window_size ${FEAT_WINDOW_SIZE} \
    --feature_size 1 \
    --one_channel 1 \
    --raw_data_paths ${DATA_PATHS} \
    --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
    --max_infill_length ${MAX_LEN_ANOMALY} \
    --ckpt_dir "${FINETUNE_CKPT}" \
    --out_dir "${OUTDIR}" \
    --generated_path "${FINETUNE_CKPT}/no_code_impute_samples.pth" \
    --gpu_id 0

cd ./NN_evaluate