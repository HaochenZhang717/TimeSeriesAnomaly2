#cd ..

export hucfg_t_sampling=logitnorm
LR=1e-4
LEN_WHOLE=600
MAX_LEN_ANOMALY=450
MIN_LEN_ANOMALY=80

ONE_CHANNEL=0
FEAT_SIZE=2

DATA_TYPE="ecg"
WANDB_PROJECT="flowts_qt_two_channels"

VQVAE_CKPT="none"
PRETRAIN_CKPT="none"
FINETUNE_CKPT="/root/tianyi/formal_experiment/qtdb_two_channels/flowts/no_code_impute_from_scratch_ckpt_lr${LR}"


DATA_PATHS='["/root/tianyi/TimeSeriesAnomaly2/dataset_utils/ECG_datasets/raw_data_qtdb/sel233.npz"]'
TEST_DATA_PATHS='["/root/tianyi/TimeSeriesAnomaly2/dataset_utils/ECG_datasets/raw_data_qtdb/sel233.npz"]'
FINETUNE_TRAIN_INDICES_PATHS='["/root/tianyi/TimeSeriesAnomaly2/dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_train.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["/root/tianyi/TimeSeriesAnomaly2/dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_test.jsonl"]'
ANOMALY_INDICES_FOR_SAMPLE='["/root/tianyi/TimeSeriesAnomaly2/dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_segments_train.jsonl"]'
NORMAL_INDICES_FOR_SAMPLE='["/root/tianyi/TimeSeriesAnomaly2/dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/normal_600.jsonl"]'
EVENT_LABELS_PATHS='["./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/event_label.npy"]'




#python dsp_flow.py \
#  --what_to_do "no_code_imputation_from_scratch" \
#  \
#  --seq_len ${LEN_WHOLE} \
#  --data_type ${DATA_TYPE} \
#  --feature_size ${FEAT_SIZE} \
#  --one_channel ${ONE_CHANNEL} \
#  --num_codes 500 \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  \
#  --raw_data_paths_train ${DATA_PATHS} \
#  --raw_data_paths_test ${TEST_DATA_PATHS} \
#  --indices_paths_train ${FINETUNE_TRAIN_INDICES_PATHS} \
#  --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
#  --indices_paths_anomaly_for_sample "[]" \
#  --min_infill_length ${MIN_LEN_ANOMALY} \
#  --max_infill_length ${MAX_LEN_ANOMALY} \
#  \
#  --lr ${LR} \
#  --batch_size 64 \
#  --max_epochs 2000 \
#  --grad_clip_norm 1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "true" \
#  --patience 500 \
#  \
#  --wandb_project ${WANDB_PROJECT} \
#  --wandb_run "no_code_impute_lr${LR}" \
#  \
#  --ckpt_dir ${FINETUNE_CKPT} \
#  --pretrained_ckpt ${PRETRAIN_CKPT} \
#  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
#  \
#  --generated_path "none" \
#  \
#  --gpu_id 0


python dsp_flow.py \
  --what_to_do "principle_no_code_impute_sample" \
  \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size ${FEAT_SIZE} \
  --one_channel ${ONE_CHANNEL} \
  --event_labels_paths_train ${EVENT_LABELS_PATHS} \
  --num_codes 500 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --raw_data_paths_train ${DATA_PATHS} \
  --raw_data_paths_test ${TEST_DATA_PATHS} \
  --indices_paths_train ${NORMAL_INDICES_FOR_SAMPLE} \
  --indices_paths_test "[]" \
  --indices_paths_anomaly_for_sample ${ANOMALY_INDICES_FOR_SAMPLE} \
  --min_infill_length ${MIN_LEN_ANOMALY} \
  --max_infill_length ${MAX_LEN_ANOMALY} \
  \
  --lr 1e-4 \
  --batch_size 64 \
  --max_epochs 2000 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project "none" \
  --wandb_run "none" \
  \
  --ckpt_dir ${FINETUNE_CKPT} \
  --pretrained_ckpt "none" \
  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
  \
  --generated_path "" \
  \
  --gpu_id 0

#python dsp_flow.py \
#  --what_to_do "no_code_impute_sample_non_downstream" \
#  \
#  --seq_len ${LEN_WHOLE} \
#  --data_type ${DATA_TYPE} \
#  --feature_size ${FEAT_SIZE} \
#  --one_channel ${ONE_CHANNEL} \
#  --num_codes 500 \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  \
#  --raw_data_paths_train ${DATA_PATHS} \
#  --raw_data_paths_test ${TEST_DATA_PATHS} \
#  --indices_paths_train ${FINETUNE_TEST_INDICES_PATHS} \
#  --indices_paths_test "[]" \
#  --indices_paths_anomaly_for_sample ${ANOMALY_INDICES_FOR_SAMPLE} \
#  --min_infill_length ${MIN_LEN_ANOMALY} \
#  --max_infill_length ${MAX_LEN_ANOMALY} \
#  \
#  --lr 1e-4 \
#  --batch_size 64 \
#  --max_epochs 2000 \
#  --grad_clip_norm 1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "true" \
#  --patience 50 \
#  \
#  --wandb_project "none" \
#  --wandb_run "none" \
#  \
#  --ckpt_dir ${FINETUNE_CKPT} \
#  --pretrained_ckpt "none" \
#  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
#  \
#  --generated_path "" \
#  \
#  --gpu_id 0



#cd ./flowts_baseline
