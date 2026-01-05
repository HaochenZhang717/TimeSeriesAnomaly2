cd ..

export hucfg_t_sampling=logitnorm
LR=1e-4
LEN_WHOLE=1000
MAX_LEN_ANOMALY=800
MIN_LEN_ANOMALY=180

ONE_CHANNEL=0
FEAT_SIZE=2

DATA_TYPE="ecg"
WANDB_PROJECT="dsp_flow_mitdb_two_channels"


DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data/106.npz"]'
TEST_DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data/106.npz"]'
PRETRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/mixed.jsonl"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_train.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_test.jsonl"]'
ANOMALY_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/anomaly_segments_with_prototype_train.jsonl"]'
NORMAL_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal_1000.jsonl"]'

#VQVAE Train Parameters
VQVAE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/mixed.jsonl"]'
CODE_DIM=8
CODE_LEN=4
NUM_CODES=500

VQVAE_CKPT="/Users/zhc/Documents/PhD/projects/TimeSeriesAnomaly/code"


python mini_runnable_vqvae.py \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "VQVAE-K${NUM_CODES}" \
  --max_seq_len ${MAX_LEN_ANOMALY} \
  --min_seq_len ${MIN_LEN_ANOMALY} \
  --data_paths ${DATA_PATHS} \
  --indices_paths ${VQVAE_TRAIN_INDICES_PATHS}\
  --data_type ${DATA_TYPE} \
  --gpu_id 0 \
  --save_dir ${VQVAE_CKPT} \
  --code_dim ${CODE_DIM} \
  --code_len ${CODE_LEN} \
  --num_codes ${NUM_CODES} \
  --one_channel ${ONE_CHANNEL} \
  --feat_size ${FEAT_SIZE}


cd dsp_our_method
