export hucfg_t_sampling=logitnorm
LR=1e-4
LEN_WHOLE=800
MAX_LEN_ANOMALY=360
MIN_LEN_ANOMALY=30
GPU_ID=4
ONE_CHANNEL=0
FEAT_SIZE=2

DATA_TYPE="ecg"
WANDB_PROJECT="dsp_flow_svdb_2_channel"

VQVAE_CKPT="/root/tianyi/formal_experiment/svdb_v_anomaly/dsp_flow/vqvae_save_path"
PRETRAIN_CKPT="/root/tianyi/formal_experiment/svdb_v_anomaly/dsp_flow/no_context_pretrain_ckpt"
FINETUNE_CKPT="/root/tianyi/formal_experiment/svdb_v_anomaly/dsp_flow/impute_finetune_ckpt_lr${LR}"


DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_svdb/859.npz"]'
TEST_DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_svdb/859.npz"]'
PRETRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/normal_360.jsonl"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/V_train.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/V_test.jsonl"]'
ANOMALY_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/V_segments_train.jsonl"]'
NORMAL_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/normal_800.jsonl"]'

#VQVAE Train Parameters
VQVAE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_svdb/slide_windows_859npz/normal_800.jsonl"]'
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


python dsp_flow.py \
  --what_to_do "no_context_pretrain" \
  \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size ${FEAT_SIZE} \
  --one_channel ${ONE_CHANNEL} \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --raw_data_paths_train ${DATA_PATHS} \
  --raw_data_paths_test ${TEST_DATA_PATHS} \
  --indices_paths_train ${PRETRAIN_INDICES_PATHS} \
  --indices_paths_test "[]" \
  --indices_paths_anomaly_for_sample "[]" \
  --min_infill_length ${MIN_LEN_ANOMALY} \
  --max_infill_length ${MAX_LEN_ANOMALY} \
  \
  --lr 1e-4 \
  --batch_size 64 \
  --max_epochs 250 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "no_context_pretrain" \
  \
  --ckpt_dir ${PRETRAIN_CKPT} \
  --pretrained_ckpt "none" \
  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt"\
  \
  --generated_path "none" \
  \
  --gpu_id ${GPU_ID}



python dsp_flow.py \
  --what_to_do "imputation_finetune" \
  \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size ${FEAT_SIZE} \
  --one_channel ${ONE_CHANNEL} \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
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
  --max_epochs 500 \
  --grad_clip_norm 1.0 \
  --grad_accum_steps 1 \
  --early_stop "true" \
  --patience 50 \
  \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run "impute_finetune_lr${LR}" \
  \
  --ckpt_dir ${FINETUNE_CKPT} \
  --pretrained_ckpt ${PRETRAIN_CKPT} \
  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
  \
  --generated_path "none" \
  \
  --gpu_id ${GPU_ID}


python dsp_flow.py \
  --what_to_do "posterior_impute_sample" \
  \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size ${FEAT_SIZE} \
  --one_channel ${ONE_CHANNEL} \
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
  --gpu_id ${GPU_ID}



python dsp_flow.py \
  --what_to_do "anomaly_evaluate" \
  \
  --seq_len ${LEN_WHOLE} \
  --data_type ${DATA_TYPE} \
  --feature_size ${FEAT_SIZE} \
  --one_channel ${ONE_CHANNEL} \
  \
  --n_layer_enc 4 \
  --n_layer_dec 4 \
  --d_model 64 \
  --n_heads 4 \
  \
  --raw_data_paths_train ${DATA_PATHS} \
  --raw_data_paths_test ${TEST_DATA_PATHS} \
  --indices_paths_train "[]" \
  --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
  --indices_paths_anomaly_for_sample "[]" \
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
  --ckpt_dir "" \
  --pretrained_ckpt "none" \
  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
  \
  --generated_path "${FINETUNE_CKPT}/posterior_impute_samples.pth" \
  \
  --gpu_id ${GPU_ID}
