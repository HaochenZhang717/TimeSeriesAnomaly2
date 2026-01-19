cd ..

LR=1e-4
LEN_WHOLE=1000
MAX_LEN_ANOMALY=800
MIN_LEN_ANOMALY=180

ONE_CHANNEL=0
FEAT_SIZE=2

DATA_TYPE="ecg"
WANDB_PROJECT="C-GATS_mitdb_two_channels"

FINETUNE_CKPT="../formal_experiment/mitdb_two_channels/C-GATS/ckpt_lr${LR}"


DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data/106.npz"]'
TEST_DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data/106.npz"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_train.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/V_test.jsonl"]'
ANOMALY_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/anomaly_segments_with_prototype_train.jsonl"]'
NORMAL_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/train/normal_1000.jsonl"]'
EVENT_LABELS_PATHS='["./dataset_utils/ECG_datasets/indices/slide_windows_106npz/event_label.npy"]'

HIDDEN_LAYER_SIZES="[50,100,200]"
TREND_POLY=3
CUSTOM_SEAS="[[10,100],[20,50],[40,25],[100,10]]"
LATENT_DIM=64
KL_WT=1e-4

#python timevae_pipeline.py \
#  --what_to_do "imputation_pretrain" \
#  \
#  --seq_len ${LEN_WHOLE} \
#  --data_type ${DATA_TYPE} \
#  --feature_size ${FEAT_SIZE} \
#  --one_channel ${ONE_CHANNEL} \
#  \
#  --hidden_layer_sizes ${HIDDEN_LAYER_SIZES} \
#  --trend_poly ${TREND_POLY} \
#  --custom_seas ${CUSTOM_SEAS} \
#  --latent_dim ${LATENT_DIM}  \
#  --kl_wt ${KL_WT} \
#  \
#  --raw_data_paths_train ${DATA_PATHS} \
#  --raw_data_paths_test ${TEST_DATA_PATHS} \
#  --event_labels_paths_train ${EVENT_LABELS_PATHS} \
#  --indices_paths_train ${NORMAL_INDICES_FOR_SAMPLE} \
#  --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
#  --indices_paths_anomaly_for_sample "[]" \
#  --min_infill_length ${MIN_LEN_ANOMALY} \
#  --max_infill_length ${MAX_LEN_ANOMALY} \
#  \
#  --lr ${LR} \
#  --batch_size 64 \
#  --max_epochs 200 \
#  --grad_clip_norm 1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "true" \
#  --patience 500 \
#  \
#  --wandb_project ${WANDB_PROJECT} \
#  --wandb_run "no_code_impute_lr${LR}" \
#  \
#  --ckpt_dir ${FINETUNE_CKPT} \
#  \
#  --generated_path "none" \
#  \
#  --gpu_id 0
#
#
#python timevae_pipeline.py \
#  --what_to_do "imputation_finetune" \
#  \
#  --seq_len ${LEN_WHOLE} \
#  --data_type ${DATA_TYPE} \
#  --feature_size ${FEAT_SIZE} \
#  --one_channel ${ONE_CHANNEL} \
#  \
#  --hidden_layer_sizes ${HIDDEN_LAYER_SIZES} \
#  --trend_poly ${TREND_POLY} \
#  --custom_seas ${CUSTOM_SEAS} \
#  --latent_dim ${LATENT_DIM}  \
#  --kl_wt 0.0 \
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
#  \
#  --generated_path "none" \
#  \
#  --gpu_id 0
#
#python timevae_pipeline.py \
#  --what_to_do "cgats_impute_sample" \
#  \
#  --seq_len ${LEN_WHOLE} \
#  --data_type ${DATA_TYPE} \
#  --feature_size ${FEAT_SIZE} \
#  --one_channel ${ONE_CHANNEL} \
#  \
#  --hidden_layer_sizes ${HIDDEN_LAYER_SIZES} \
#  --trend_poly ${TREND_POLY} \
#  --custom_seas ${CUSTOM_SEAS} \
#  --latent_dim ${LATENT_DIM}  \
#  --kl_wt ${KL_WT} \
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
#  \
#  --generated_path "" \
#  \
#  --gpu_id 0



python timevae_pipeline.py \
  --what_to_do "cgats_impute_sample_non_downstream" \
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
  \
  --generated_path "" \
  \
  --gpu_id 0


#OUTDIR="../nn_eval/mitdb_two_channels/C-GATS"
#
#
#python run_nn_evaluate.py \
#    --seq_len ${LEN_WHOLE} \
#    --feature_size 2 \
#    --one_channel 0 \
#    --feat_window_size 300 \
#    --raw_data_paths ${DATA_PATHS} \
#    --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
#    --max_infill_length ${MAX_LEN_ANOMALY} \
#    --ckpt_dir "${FINETUNE_CKPT}" \
#    --out_dir "${OUTDIR}" \
#    --generated_path "${FINETUNE_CKPT}/no_code_impute_samples.pth" \
#    --gpu_id 0

cd timevae_all

