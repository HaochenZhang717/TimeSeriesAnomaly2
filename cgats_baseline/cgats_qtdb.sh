cd ..

PRETRAIN_CKPT_DIR="/root/tianyi/formal_experiment/qtdb_two_channels/cgats/pretrain_ckpt"
FINETUNE_CKPT_DIR="/root/tianyi/formal_experiment/qtdb_two_channels/cgats/finetune_ckpt"

LEN_WHOLE=600
MAX_LEN_ANOMALY=450
MIN_LEN_ANOMALY=80
FEATURE_SIZE=2
ONE_CHANNEL=0

RAW_DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_qtdb/sel233.npz"]'
PRETRAIN_RAW_DATA_PATHS="./dataset_utils/ECG_datasets/raw_data_qtdb/sel233.npz"
PRETRAIN_INDICES_PATHS_TRAIN="./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/normal_600.jsonl"
INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/normal_600.jsonl"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_train.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_test.jsonl"]'


WANDB_PROJECT="CGATS-QTDB"

python CGATSPretrainPipeline.py \
  --seq_len ${LEN_WHOLE} \
  --feature_size ${FEATURE_SIZE} \
  --one_channel ${ONE_CHANNEL} \
  \
  --latent_dim 64 \
  --trend_poly 3 \
  --kl_wt 1e-3 \
  --hidden_layer_sizes "[50,100,200]" \
  --custom_seas "[[10,60],[20,30],[40,15],[60,10]]" \
  \
  --max_anomaly_length ${MAX_LEN_ANOMALY} \
  --min_anomaly_length ${MIN_LEN_ANOMALY} \
  --raw_data_paths_train ${PRETRAIN_RAW_DATA_PATHS} \
  --indices_paths_train ${PRETRAIN_INDICES_PATHS_TRAIN} \
  \
  --lr 1e-5 \
  --batch_size 128 \
  --epochs 100 \
  --grad_clip_norm 1.0 \
  --early_stop "true" \
  --patience 200 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "pretrain" \
  \
  --ckpt_dir ${PRETRAIN_CKPT_DIR} \
  --gpu_id 0



python CGATSFinetunePipeline.py \
  --what_to_do "finetune" \
  \
  --seq_len ${LEN_WHOLE} \
  --feature_size ${FEATURE_SIZE} \
  --one_channel ${ONE_CHANNEL} \
  \
  --latent_dim 64 \
  --trend_poly 3 \
  --kl_wt 1e-3 \
  --hidden_layer_sizes "[50,100,200]" \
  --custom_seas "[[10,60],[20,30],[40,15],[60,10]]" \
  \
  --max_anomaly_length ${MAX_LEN_ANOMALY} \
  --min_anomaly_length ${MIN_LEN_ANOMALY} \
  --raw_data_paths_train ${RAW_DATA_PATHS} \
  --raw_data_paths_test ${RAW_DATA_PATHS} \
  --indices_paths_train ${FINETUNE_TRAIN_INDICES_PATHS} \
  --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
  \
  --lr 1e-5 \
  --batch_size 64 \
  --epochs 500 \
  --grad_clip_norm 1.0 \
  --early_stop "true" \
  --patience 500 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "finetune" \
  \
  --ckpt_dir ${FINETUNE_CKPT_DIR} \
  --pretrained_ckpt "${PRETRAIN_CKPT_DIR}/ckpt.pth" \
  \
  --gpu_id 0


python CGATSFinetunePipeline.py \
  --what_to_do "sample_anomaly" \
  \
  --seq_len ${LEN_WHOLE} \
  --feature_size ${FEATURE_SIZE} \
  --one_channel ${ONE_CHANNEL} \
  \
  --latent_dim 64 \
  --trend_poly 3 \
  --kl_wt 1e-3 \
  --hidden_layer_sizes "[50,100,200]" \
  --custom_seas "[[10,60],[20,30],[40,15],[60,10]]" \
  \
  --max_anomaly_length ${MAX_LEN_ANOMALY} \
  --min_anomaly_length ${MIN_LEN_ANOMALY} \
  --raw_data_paths_train ${RAW_DATA_PATHS} \
  --raw_data_paths_test ${RAW_DATA_PATHS} \
  --indices_paths_train ${INDICES_FOR_SAMPLE} \
  --indices_paths_test ${INDICES_FOR_SAMPLE} \
  \
  --lr 1e-5 \
  --batch_size 128 \
  --epochs 20 \
  --grad_clip_norm 1.0 \
  --early_stop "true" \
  --patience 500 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "finetune" \
  \
  --ckpt_dir ${FINETUNE_CKPT_DIR} \
  --pretrained_ckpt "none" \
  \
  --gpu_id 0




python CGATSFinetunePipeline.py \
  --what_to_do "sample_anomaly_non_downstream" \
  \
  --seq_len ${LEN_WHOLE} \
  --feature_size ${FEATURE_SIZE} \
  --one_channel ${ONE_CHANNEL} \
  \
  --latent_dim 64 \
  --trend_poly 3 \
  --kl_wt 1e-3 \
  --hidden_layer_sizes "[50,100,200]" \
  --custom_seas "[[10,60],[20,30],[40,15],[60,10]]" \
  \
  --max_anomaly_length ${MAX_LEN_ANOMALY} \
  --min_anomaly_length ${MIN_LEN_ANOMALY} \
  --raw_data_paths_train ${RAW_DATA_PATHS} \
  --raw_data_paths_test ${RAW_DATA_PATHS} \
  --indices_paths_train ${FINETUNE_TEST_INDICES_PATHS} \
  --indices_paths_test ${INDICES_FOR_SAMPLE} \
  \
  --lr 1e-5 \
  --batch_size 128 \
  --epochs 20 \
  --grad_clip_norm 1.0 \
  --early_stop "true" \
  --patience 500 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "finetune" \
  \
  --ckpt_dir ${FINETUNE_CKPT_DIR} \
  --pretrained_ckpt "none" \
  \
  --gpu_id 0


cd ./cgats_baseline

