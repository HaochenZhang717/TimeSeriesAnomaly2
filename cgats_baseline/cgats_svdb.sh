cd ..

GPU_ID=0

PRETRAIN_CKPT_DIR="/root/tianyi/formal_experiment/svdb_one_channel/cgats/pretrain_ckpt"
LEN_WHOLE=800
MAX_LEN_ANOMALY=360
MIN_LEN_ANOMALY=30
FEATURE_SIZE=1
ONE_CHANNEL=1

RAW_DATA_PATHS="./dataset_utils/ECG_datasets/raw_data_svdb/859.npz"

PRETRAIN_INDICES_PATHS_TRAIN="./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/normal_800.jsonl"

WANDB_PROJECT="CGATS-SVDB"

python CGATSPretrainPipeline.py \
  --seq_len ${LEN_WHOLE} \
  --feature_size ${FEATURE_SIZE} \
  --on_channel ${ONE_CHANNEL} \
  \
  --latent_dim 64 \
  --trend_poly 3 \
  --kl_wt 1e-3 \
  --hidden_layer_sizes "[50,100,200]" \
  --custom_seas "[[10,80],[20,40],[40,20],[80,10]]" \
  \
  --max_anomaly_length ${MAX_LEN_ANOMALY} \
  --min_anomaly_length ${MIN_LEN_ANOMALY} \
  --raw_data_paths_train ${RAW_DATA_PATHS} \
  --indices_paths_train ${PRETRAIN_INDICES_PATHS_TRAIN} \
  \
  --lr 1e-5 \
  --batch_size 128 \
  --epochs 2000 \
  --grad_clip_norm 1.0 \
  --early_stop "true" \
  --patience 500 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run pretrain \
  \
  --ckpt_dir ${PRETRAIN_CKPT_DIR} \
  --gpu_id ${GPU_ID}



python CGATSFinetunePipeline.py \
  --what_to_do "finetune" \
  \
  --seq_len ${LEN_WHOLE} \
  --feature_size ${FEATURE_SIZE} \
  --on_channel ${ONE_CHANNEL} \
  \
  --latent_dim 64 \
  --trend_poly 3 \
  --kl_wt 1e-3 \
  --hidden_layer_sizes "[50,100,200]" \
  --custom_seas "[[10,80],[20,40],[40,20],[80,10]]" \
  \
  --max_anomaly_length ${MAX_LEN_ANOMALY} \
  --min_anomaly_length ${MIN_LEN_ANOMALY} \
  --raw_data_paths_train ${RAW_DATA_PATHS} \
  --normal_indices_paths_train  \
  --anomaly_indices_paths_train \
  \
  --lr 1e-5 \
  --batch_size 128 \
  --epochs 2000 \
  --grad_clip_norm 1.0 \
  --early_stop "true" \
  --patience 500 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run pretrain \
  \
  --ckpt_dir ${PRETRAIN_CKPT_DIR} \
  --gpu_id ${GPU_ID}


cd ./cgats_baseline

