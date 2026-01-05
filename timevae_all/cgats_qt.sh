
LR=1e-4
LEN_WHOLE=600
MAX_LEN_ANOMALY=450
MIN_LEN_ANOMALY=80

ONE_CHANNEL=0
FEAT_SIZE=2

DATA_TYPE="ecg"
WANDB_PROJECT="C-GATS_qt_two_channels"


FINETUNE_CKPT="/root/tianyi/formal_experiment/qtdb_two_channels/C-GATS/ckpt_lr${LR}"


DATA_PATHS='["/root/tianyi/TimeSeriesAnomaly2/dataset_utils/ECG_datasets/raw_data_qtdb/sel233.npz"]'
TEST_DATA_PATHS='["/root/tianyi/TimeSeriesAnomaly2/dataset_utils/ECG_datasets/raw_data_qtdb/sel233.npz"]'
FINETUNE_TRAIN_INDICES_PATHS='["/root/tianyi/TimeSeriesAnomaly2/dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_train.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["/root/tianyi/TimeSeriesAnomaly2/dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_test.jsonl"]'
ANOMALY_INDICES_FOR_SAMPLE='["/root/tianyi/TimeSeriesAnomaly2/dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_segments_train.jsonl"]'
NORMAL_INDICES_FOR_SAMPLE='["/root/tianyi/TimeSeriesAnomaly2/dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/normal_600.jsonl"]'

HIDDEN_LAYER_SIZES="[50,100,200]"
TREND_POLY=3
CUSTOM_SEAS="[[10,60],[20,30],[40,15],[60,10]]"
LATENT_DIM=64
KL_WT=1e-3



python timevae_pipeline.py \
  --what_to_do "imputation_pretrain" \
  \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size ${FEAT_SIZE} \
  --one_channel ${ONE_CHANNEL} \
  \
  --hidden_layer_sizes ${HIDDEN_LAYER_SIZES} \
  --trend_poly ${TREND_POLY} \
  --custom_seas ${CUSTOM_SEAS} \
  --latent_dim ${LATENT_DIM}  \
  --kl_wt ${KL_WT} \
  \
  --raw_data_paths_train ${DATA_PATHS} \
  --raw_data_paths_test ${TEST_DATA_PATHS} \
  --indices_paths_train ${NORMAL_INDICES_FOR_SAMPLE} \
  --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
  --indices_paths_anomaly_for_sample "[]" \
  --min_infill_length ${MIN_LEN_ANOMALY} \
  --max_infill_length ${MAX_LEN_ANOMALY} \
  \
  --lr ${LR} \
  --batch_size 64 \
  --max_epochs 2000 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 500 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "no_code_impute_lr${LR}" \
  \
  --ckpt_dir ${FINETUNE_CKPT} \
  \
  --generated_path "none" \
  \
  --gpu_id 0


python timevae_pipeline.py \
  --what_to_do "imputation_finetune" \
  \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size ${FEAT_SIZE} \
  --one_channel ${ONE_CHANNEL} \
  \
  --hidden_layer_sizes ${HIDDEN_LAYER_SIZES} \
  --trend_poly ${TREND_POLY} \
  --custom_seas ${CUSTOM_SEAS} \
  --latent_dim ${LATENT_DIM}  \
  --kl_wt ${KL_WT} \
  \
  --raw_data_paths_train ${DATA_PATHS} \
  --raw_data_paths_test ${TEST_DATA_PATHS} \
  --indices_paths_train ${FINETUNE_TRAIN_INDICES_PATHS} \
  --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
  --indices_paths_anomaly_for_sample "[]" \
  --min_infill_length ${MIN_LEN_ANOMALY} \
  --max_infill_length ${MAX_LEN_ANOMALY} \
  \
  --lr ${LR} \
  --batch_size 64 \
  --max_epochs 2000 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 500 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "no_code_impute_lr${LR}" \
  \
  --ckpt_dir ${FINETUNE_CKPT} \
  \
  --generated_path "none" \
  \
  --gpu_id 0

