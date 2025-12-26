export hucfg_t_sampling=logitnorm
LR=1e-4
LEN_WHOLE=800
MAX_LEN_ANOMALY=450
MIN_LEN_ANOMALY=80
GPU_ID=6
ONE_CHANNEL=0
FEAT_SIZE=2

DATA_TYPE="ecg"
WANDB_PROJECT="dsp_flow_incart_3in1"

VQVAE_CKPT="/Users/zhc/Documents/PhD/projects/TimeSeriesAnomaly/incart_results"
PRETRAIN_CKPT="/root/tianyi/formal_experiment/incart_3in1/dsp_flow/no_context_pretrain_ckpt"
FINETUNE_CKPT="/root/tianyi/formal_experiment/incart_3in1/dsp_flow/impute_finetune_ckpt_lr${LR}"


DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_incart/I03.npz","./dataset_utils/ECG_datasets/raw_data_incart/I52.npz","./dataset_utils/ECG_datasets/raw_data_incart/I69.npz"]'
TEST_DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_incart/I03.npz","./dataset_utils/ECG_datasets/raw_data_incart/I52.npz","./dataset_utils/ECG_datasets/raw_data_incart/I69.npz"]'
PRETRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_incart/slide_windows_I03npz/normal_450.jsonl","./dataset_utils/ECG_datasets/indices_incart/slide_windows_I52npz/normal_450.jsonl","./dataset_utils/ECG_datasets/indices_incart/slide_windows_I69npz/normal_450.jsonl"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_incart/slide_windows_I03npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_incart/slide_windows_I52npz/V_train.jsonl","./dataset_utils/ECG_datasets/indices_incart/slide_windows_I69npz/V_train.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_incart/slide_windows_I03npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_incart/slide_windows_I52npz/V_test.jsonl","./dataset_utils/ECG_datasets/indices_incart/slide_windows_I69npz/V_test.jsonl"]'
ANOMALY_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices_incart/slide_windows_I03npz/V_segments_train.jsonl","./dataset_utils/ECG_datasets/indices_incart/slide_windows_I52npz/V_segments_train.jsonl","./dataset_utils/ECG_datasets/indices_incart/slide_windows_I69npz/V_segments_train.jsonl"]'
NORMAL_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices_incart/slide_windows_I03npz/normal_800.jsonl","./dataset_utils/ECG_datasets/indices_incart/slide_windows_I52npz/normal_800.jsonl","./dataset_utils/ECG_datasets/indices_incart/slide_windows_I69npz/normal_800.jsonl"]'

#VQVAE Train Parameters
VQVAE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_incart/slide_windows_I03npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_incart/slide_windows_I52npz/mixed.jsonl","./dataset_utils/ECG_datasets/indices_incart/slide_windows_I69npz/mixed.jsonl"]'
CODE_DIM=8
CODE_LEN=4
NUM_CODES=500


python mini_runnable_vqvae.py \
  --max_seq_len ${MAX_LEN_ANOMALY} \
  --min_seq_len ${MIN_LEN_ANOMALY} \
  --data_paths ${DATA_PATHS} \
  --indices_paths ${VQVAE_TRAIN_INDICES_PATHS}\
  --data_type ${DATA_TYPE} \
  --gpu_id ${GPU_ID} \
  --save_dir ${VQVAE_CKPT} \
  --code_dim ${CODE_DIM} \
  --code_len ${CODE_LEN} \
  --num_codes ${NUM_CODES} \
  --one_channel ${ONE_CHANNEL} \
  --feat_size ${FEAT_SIZE}

