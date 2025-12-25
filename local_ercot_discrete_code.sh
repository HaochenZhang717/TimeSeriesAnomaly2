export hucfg_t_sampling=logitnorm
LR=1e-4
LEN_WHOLE=1000
MAX_LEN_ANOMALY=200
MIN_LEN_ANOMALY=190
GPU_ID=3

DATA_TYPE="ercot"
WANDB_PROJECT="dsp_flow_ercot"

VQVAE_CKPT="/Users/zhc/Documents/PhD/projects/TimeSeriesAnomaly/ercot/dsp_flow/vqvae_save_path"

DATA_PATHS='["./dataset_utils/ERCOT_datasets/raw_data/coast.npy", "./dataset_utils/ERCOT_datasets/raw_data/east.npy", "./dataset_utils/ERCOT_datasets/raw_data/fwest.npy", "./dataset_utils/ERCOT_datasets/raw_data/ncent.npy"]'
TEST_DATA_PATHS='["./dataset_utils/ERCOT_datasets/raw_data/north.npy", "./dataset_utils/ERCOT_datasets/raw_data/scent.npy", "./dataset_utils/ERCOT_datasets/raw_data/south.npy", "./dataset_utils/ERCOT_datasets/raw_data/west.npy"]'
PRETRAIN_INDICES_PATHS='["./dataset_utils/ERCOT_datasets/indices/coast_normal_200.jsonl", "./dataset_utils/ERCOT_datasets/indices/east_normal_200.jsonl", "./dataset_utils/ERCOT_datasets/indices/fwest_normal_200.jsonl", "./dataset_utils/ERCOT_datasets/indices/ncent_normal_200.jsonl"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ERCOT_datasets/indices/coast_anomaly.jsonl", "./dataset_utils/ERCOT_datasets/indices/east_anomaly.jsonl", "./dataset_utils/ERCOT_datasets/indices/fwest_anomaly.jsonl", "./dataset_utils/ERCOT_datasets/indices/ncent_anomaly.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ERCOT_datasets/indices/north_anomaly.jsonl", "./dataset_utils/ERCOT_datasets/indices/scent_anomaly.jsonl", "./dataset_utils/ERCOT_datasets/indices/south_anomaly.jsonl", "./dataset_utils/ERCOT_datasets/indices/west_anomaly.jsonl"]'


ANOMALY_INDICES_FOR_SAMPLE='["./dataset_utils/ERCOT_datasets/indices/anomaly_segments_coast.jsonl", "./dataset_utils/ERCOT_datasets/indices/anomaly_segments_east.jsonl", "./dataset_utils/ERCOT_datasets/indices/anomaly_segments_fwest.jsonl", "./dataset_utils/ERCOT_datasets/indices/anomaly_segments_ncent.jsonl"]'
NORMAL_INDICES_FOR_SAMPLE='["./dataset_utils/ERCOT_datasets/indices/coast_normal_1000.jsonl", "./dataset_utils/ERCOT_datasets/indices/east_normal_1000.jsonl", "./dataset_utils/ERCOT_datasets/indices/fwest_normal_1000.jsonl", "./dataset_utils/ERCOT_datasets/indices/ncent_normal_1000.jsonl"]'

#VQVAE Train Parameters
VQVAE_TRAIN_INDICES_PATHS='["./dataset_utils/ERCOT_datasets/indices/coast_normal_200.jsonl", "./dataset_utils/ERCOT_datasets/indices/east_normal_200.jsonl", "./dataset_utils/ERCOT_datasets/indices/fwest_normal_200.jsonl", "./dataset_utils/ERCOT_datasets/indices/ncent_normal_200.jsonl"]'
CODE_DIM=8
CODE_LEN=4
NUM_CODES=500

python mini_runnable_vqvae.py \
  --max_seq_len ${MAX_LEN_ANOMALY} \
  --min_seq_len ${MIN_LEN_ANOMALY} \
  --data_paths "${DATA_PATHS}" \
  --indices_paths "${VQVAE_TRAIN_INDICES_PATHS}"\
  --data_type ${DATA_TYPE} \
  --gpu_id ${GPU_ID} \
  --save_dir ${VQVAE_CKPT} \
  --code_dim ${CODE_DIM} \
  --code_len ${CODE_LEN} \
  --num_codes ${NUM_CODES}
