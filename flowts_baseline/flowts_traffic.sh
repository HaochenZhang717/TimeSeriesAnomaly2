cd ..

export hucfg_t_sampling=logitnorm
LR=1e-4
LEN_WHOLE=72
MAX_LEN_ANOMALY=24
MIN_LEN_ANOMALY=23

ONE_CHANNEL=1
FEAT_SIZE=1

DATA_TYPE="ecg"
WANDB_PROJECT="flowts_traffic"

VQVAE_CKPT="none"
PRETRAIN_CKPT="none"
FINETUNE_CKPT="../formal_experiment/traffic/flowts/no_code_impute_from_scratch_ckpt_lr${LR}"



DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_traffic/metro_traffic_data.npz"]'
TEST_DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_traffic/metro_traffic_data.npz"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_traffic/slide_windows_metro_traffic_datanpz/V_train.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_traffic/slide_windows_metro_traffic_datanpz/V_test.jsonl"]'
ANOMALY_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices_traffic/slide_windows_metro_traffic_datanpz/V_segments_train.jsonl"]'
NORMAL_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices_traffic/slide_windows_metro_traffic_datanpz/normal_72.jsonl"]'
EVENT_LABELS_PATHS='["./dataset_utils/ECG_datasets/indices_traffic/slide_windows_metro_traffic_datanpz/event_label.npy"]'


#
#
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
#  --event_labels_paths_train ${EVENT_LABELS_PATHS} \
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
#
#
#python dsp_flow.py \
#  --what_to_do "principle_no_code_impute_sample" \
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
#  --event_labels_paths_train ${EVENT_LABELS_PATHS} \
#  --indices_paths_train ${NORMAL_INDICES_FOR_SAMPLE} \
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

python dsp_flow.py \
  --what_to_do "no_code_impute_sample_non_downstream" \
  \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size ${FEAT_SIZE} \
  --one_channel ${ONE_CHANNEL} \
  --num_codes 500 \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --raw_data_paths_train ${DATA_PATHS} \
  --raw_data_paths_test ${TEST_DATA_PATHS} \
  --event_labels_paths_train ${EVENT_LABELS_PATHS} \
  --indices_paths_train ${FINETUNE_TRAIN_INDICES_PATHS} \
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

#OUTDIR="../nn_eval/traffic/flowts"
#
#
#python run_nn_evaluate.py \
#    --seq_len ${LEN_WHOLE} \
#    --feature_size 1 \
#    --one_channel 1 \
#    --feat_window_size 50 \
#    --raw_data_paths ${DATA_PATHS} \
#    --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
#    --max_infill_length ${MAX_LEN_ANOMALY} \
#    --ckpt_dir "${FINETUNE_CKPT}" \
#    --out_dir "${OUTDIR}" \
#    --generated_path "${FINETUNE_CKPT}/principle_no_code_impute_samples.pth" \
#    --gpu_id 0


cd flowts_baseline

