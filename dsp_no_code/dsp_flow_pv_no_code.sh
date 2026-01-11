export hucfg_t_sampling=logitnorm
cd ..

LR=1e-4
LEN_WHOLE=200
MAX_LEN_ANOMALY=144
MIN_LEN_ANOMALY=20

ONE_CHANNEL=1
FEAT_SIZE=1

DATA_TYPE="ecg"
WANDB_PROJECT="dsp_flow_PV"

VQVAE_CKPT="none"
PRETRAIN_CKPT="../formal_experiment/PV/dsp_flow_no_code/no_context_no_code_pretrain_ckpt"
FINETUNE_CKPT="../formal_experiment/PV/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr${LR}"



DATA_PATHS='[
    "./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_2.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_3.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_4.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2015_pv_sub_0.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2015_pv_sub_1.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2021_pv_live_0.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2022_pv_live_0.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2023_pv_live_0.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2024_pv_live_0.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2025_pv_live_0.npz"
]'

TEST_DATA_PATHS='[
    "./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_2.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_3.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2013_pv_sub_4.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2015_pv_sub_0.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2015_pv_sub_1.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2021_pv_live_0.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2022_pv_live_0.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2023_pv_live_0.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2024_pv_live_0.npz",
    "./dataset_utils/ECG_datasets/raw_data_PV/2025_pv_live_0.npz"
]'

PRETRAIN_INDICES_PATHS='[
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/mixed.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/mixed.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/mixed.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/mixed.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/mixed.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/mixed.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/mixed.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/mixed.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/mixed.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/mixed.jsonl"
]'

FINETUNE_TRAIN_INDICES_PATHS='[
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/V_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/V_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/V_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/V_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/V_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/V_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/V_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/V_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/V_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/V_train.jsonl"
]'

FINETUNE_TEST_INDICES_PATHS='[
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/V_test.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/V_test.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/V_test.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/V_test.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/V_test.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/V_test.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/V_test.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/V_test.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/V_test.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/V_test.jsonl"
]'

ANOMALY_INDICES_FOR_SAMPLE='[
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/V_segments_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/V_segments_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/V_segments_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/V_segments_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/V_segments_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/V_segments_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/V_segments_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/V_segments_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/V_segments_train.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/V_segments_train.jsonl"
]'


NORMAL_INDICES_FOR_SAMPLE='[
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/normal_200.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/normal_200.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/normal_200.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/normal_200.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/normal_200.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/normal_200.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/normal_200.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/normal_200.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/normal_200.jsonl",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/normal_200.jsonl"
]'


EVENT_LABELS_PATHS='[
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_2npz/event_label.npy",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_3npz/event_label.npy",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2013_pv_sub_4npz/event_label.npy",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_0npz/event_label.npy",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2015_pv_sub_1npz/event_label.npy",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2021_pv_live_0npz/event_label.npy",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2022_pv_live_0npz/event_label.npy",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2023_pv_live_0npz/event_label.npy",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2024_pv_live_0npz/event_label.npy",
    "./dataset_utils/ECG_datasets/indices_PV/slide_windows_2025_pv_live_0npz/event_label.npy"
]'




python dsp_flow.py \
  --what_to_do "no_context_no_code_pretrain" \
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
  --indices_paths_train ${PRETRAIN_INDICES_PATHS} \
  --indices_paths_test "[]" \
  --indices_paths_anomaly_for_sample "[]" \
  --min_infill_length ${MIN_LEN_ANOMALY} \
  --max_infill_length ${MAX_LEN_ANOMALY} \
  \
  --lr 1e-4 \
  --batch_size 64 \
  --max_epochs 100 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "no_context_no_code_pretrain" \
  \
  --ckpt_dir ${PRETRAIN_CKPT} \
  --pretrained_ckpt "none" \
  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt"\
  \
  --generated_path "none" \
  \
  --gpu_id 0



python dsp_flow.py \
  --what_to_do "no_code_imputation_finetune" \
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
  --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
  --indices_paths_anomaly_for_sample "[]" \
  --min_infill_length ${MIN_LEN_ANOMALY} \
  --max_infill_length ${MAX_LEN_ANOMALY} \
  \
  --lr ${LR} \
  --batch_size 64 \
  --max_epochs 500 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 500 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "no_code_impute_finetune_lr${LR}" \
  \
  --ckpt_dir ${FINETUNE_CKPT} \
  --pretrained_ckpt ${PRETRAIN_CKPT} \
  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
  \
  --generated_path "none" \
  \
  --gpu_id 0


python dsp_flow.py \
  --what_to_do "principle_no_code_impute_sample" \
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
  --indices_paths_train ${FINETUNE_TEST_INDICES_PATHS} \
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

cd  ./dsp_no_code