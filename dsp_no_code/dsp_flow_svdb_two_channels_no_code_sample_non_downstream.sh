cd ..


export hucfg_t_sampling=logitnorm
LR=1e-4
LEN_WHOLE=800
MAX_LEN_ANOMALY=360
MIN_LEN_ANOMALY=30
GPU_ID=0
ONE_CHANNEL=0
FEAT_SIZE=2

DATA_TYPE="ecg"
WANDB_PROJECT="dsp_flow_svdb_two_channels"

VQVAE_CKPT="none"
PRETRAIN_CKPT="/root/tianyi/formal_experiment/svdb_two_channels/dsp_flow_no_code/no_context_no_code_pretrain_ckpt"
FINETUNE_CKPT="/root/tianyi/formal_experiment/svdb_two_channels/dsp_flow_no_code/no_code_impute_finetune_ckpt_lr${LR}"


DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_svdb/859.npz"]'
TEST_DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_svdb/859.npz"]'
PRETRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/normal_360.jsonl"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/V_train.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/V_test.jsonl"]'
ANOMALY_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/V_segments_train.jsonl"]'
NORMAL_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/normal_800.jsonl"]'

echo "svdb sample start."


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


echo "svdb sample end."


cd  ./dsp_no_code