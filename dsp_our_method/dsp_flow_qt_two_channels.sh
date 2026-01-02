cd ..
export hucfg_t_sampling=logitnorm
LR=1e-4
LEN_WHOLE=600
MAX_LEN_ANOMALY=450
MIN_LEN_ANOMALY=80
#GPU_ID=0
ONE_CHANNEL=0
FEAT_SIZE=2

DATA_TYPE="ecg"
WANDB_PROJECT="dsp_flow_qt_two_channels"

#VQVAE_CKPT="/root/tianyi/formal_experiment/qtdb_two_channels/dsp_flow/vqvae_save_path"
#PRETRAIN_CKPT="/root/tianyi/formal_experiment/qtdb_two_channels/dsp_flow/no_context_pretrain_ckpt"
#FINETUNE_CKPT="/root/tianyi/formal_experiment/qtdb_two_channels/dsp_flow/impute_finetune_ckpt_lr${LR}"

#/root/tianyi/formal_experiment/qtdb/dsp_flow/impute_finetune_ckpt_lr1e-4
DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_qtdb/sel233.npz"]'
TEST_DATA_PATHS='["./dataset_utils/ECG_datasets/raw_data_qtdb/sel233.npz"]'
PRETRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/normal_450.jsonl"]'
FINETUNE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_train.jsonl"]'
FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_test.jsonl"]'
#FINETUNE_TEST_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_test.jsonl"]'
ANOMALY_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/V_segments_train.jsonl"]'
NORMAL_INDICES_FOR_SAMPLE='["./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/normal_600.jsonl"]'

#VQVAE Train Parameters
VQVAE_TRAIN_INDICES_PATHS='["./dataset_utils/ECG_datasets/indices_qtdb/slide_windows_sel233npz/normal_450.jsonl"]'

CODE_DIM=8
CODE_LEN=4
#NUM_CODES=500


GPU_IDS=(4 5 6 7)
NUM_CODES_LIST=(200 300 400 500)


for i in ${!GPU_IDS[@]}; do
(
  set -e

  GPU=${GPU_IDS[$i]}
  NUM_CODES=${NUM_CODES_LIST[$i]}
  export CUDA_VISIBLE_DEVICES=${GPU}

  VQVAE_CKPT="/root/tianyi/formal_experiment/qtdb_two_channels/dsp_flow_K${NUM_CODES}/vqvae_save_path"
  PRETRAIN_CKPT="/root/tianyi/formal_experiment/qtdb_two_channels/dsp_flow_K${NUM_CODES}/no_context_pretrain_ckpt"
  FINETUNE_CKPT="/root/tianyi/formal_experiment/qtdb_two_channels/dsp_flow_K${NUM_CODES}/impute_finetune_ckpt_lr${LR}"

  echo "Launching NUM_CODES=${NUM_CODES} on GPU ${GPU}"

  python mini_runnable_vqvae.py \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run "VQVAE-K${NUM_CODES}" \
    --max_seq_len ${MAX_LEN_ANOMALY} \
    --min_seq_len ${MIN_LEN_ANOMALY} \
    --data_paths ${DATA_PATHS} \
    --indices_paths ${VQVAE_TRAIN_INDICES_PATHS} \
    --data_type ${DATA_TYPE} \
    --gpu_id 0 \
    --save_dir ${VQVAE_CKPT} \
    --code_dim ${CODE_DIM} \
    --code_len ${CODE_LEN} \
    --num_codes ${NUM_CODES} \
    --one_channel ${ONE_CHANNEL} \
    --feat_size ${FEAT_SIZE}

  python dsp_flow.py \
    --what_to_do "no_context_pretrain" \
    --num_codes ${NUM_CODES} \
    --seq_len ${LEN_WHOLE} \
    --data_type ${DATA_TYPE} \
    --feature_size ${FEAT_SIZE} \
    --one_channel ${ONE_CHANNEL} \
    --n_layer_enc 4 \
    --n_layer_dec 4 \
    --d_model 64 \
    --n_heads 4 \
    --raw_data_paths_train ${DATA_PATHS} \
    --raw_data_paths_test ${TEST_DATA_PATHS} \
    --indices_paths_train ${PRETRAIN_INDICES_PATHS} \
    --indices_paths_test "[]" \
    --indices_paths_anomaly_for_sample "[]" \
    --min_infill_length ${MIN_LEN_ANOMALY} \
    --max_infill_length ${MAX_LEN_ANOMALY} \
    --lr 1e-4 \
    --batch_size 64 \
    --max_epochs 100 \
    --grad_clip_norm 1.0 \
    --grad_accum_steps 1 \
    --early_stop "true" \
    --patience 50 \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run "no_context_pretrain_K${NUM_CODES}" \
    --ckpt_dir ${PRETRAIN_CKPT} \
    --pretrained_ckpt "none" \
    --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
    --generated_path "none" \
    --gpu_id 0

  python dsp_flow.py \
    --what_to_do "imputation_finetune" \
    --num_codes ${NUM_CODES} \
    --seq_len ${LEN_WHOLE} \
    --data_type ${DATA_TYPE} \
    --feature_size ${FEAT_SIZE} \
    --one_channel ${ONE_CHANNEL} \
    --n_layer_enc 4 \
    --n_layer_dec 4 \
    --d_model 64 \
    --n_heads 4 \
    --raw_data_paths_train ${DATA_PATHS} \
    --raw_data_paths_test ${TEST_DATA_PATHS} \
    --indices_paths_train ${FINETUNE_TRAIN_INDICES_PATHS} \
    --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
    --indices_paths_anomaly_for_sample "[]" \
    --min_infill_length ${MIN_LEN_ANOMALY} \
    --max_infill_length ${MAX_LEN_ANOMALY} \
    --lr ${LR} \
    --batch_size 64 \
    --max_epochs 500 \
    --grad_clip_norm 1.0 \
    --grad_accum_steps 1 \
    --early_stop "true" \
    --patience 500 \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run "impute_finetune_lr${LR}_K${NUM_CODES}" \
    --ckpt_dir ${FINETUNE_CKPT} \
    --pretrained_ckpt ${PRETRAIN_CKPT} \
    --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
    --generated_path "none" \
    --gpu_id 0

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
    --gpu_id 0

) &
done

wait
echo "All jobs finished."



#python mini_runnable_vqvae.py \
#  --max_seq_len ${MAX_LEN_ANOMALY} \
#  --min_seq_len ${MIN_LEN_ANOMALY} \
#  --data_paths ${DATA_PATHS} \
#  --indices_paths ${VQVAE_TRAIN_INDICES_PATHS}\
#  --data_type ${DATA_TYPE} \
#  --gpu_id ${GPU_ID} \
#  --save_dir ${VQVAE_CKPT} \
#  --code_dim ${CODE_DIM} \
#  --code_len ${CODE_LEN} \
#  --num_codes ${NUM_CODES} \
#  --one_channel ${ONE_CHANNEL} \
#  --feat_size ${FEAT_SIZE}
#
#
#python dsp_flow.py \
#  --what_to_do "no_context_pretrain" \
#  \
#  --seq_len ${LEN_WHOLE} \
#  --data_type ${DATA_TYPE} \
#  --feature_size ${FEAT_SIZE} \
#  --one_channel ${ONE_CHANNEL} \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  \
#  --raw_data_paths_train ${DATA_PATHS} \
#  --raw_data_paths_test ${TEST_DATA_PATHS} \
#  --indices_paths_train ${PRETRAIN_INDICES_PATHS} \
#  --indices_paths_test "[]" \
#  --indices_paths_anomaly_for_sample "[]" \
#  --min_infill_length ${MIN_LEN_ANOMALY} \
#  --max_infill_length ${MAX_LEN_ANOMALY} \
#  \
#  --lr 1e-4 \
#  --batch_size 64 \
#  --max_epochs 100 \
#  --grad_clip_norm 1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "true" \
#  --patience 50 \
#  \
#  --wandb_project ${WANDB_PROJECT} \
#  --wandb_run "no_context_pretrain" \
#  \
#  --ckpt_dir ${PRETRAIN_CKPT} \
#  --pretrained_ckpt "none" \
#  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt"\
#  \
#  --generated_path "none" \
#  \
#  --gpu_id ${GPU_ID}
#
#
#
#python dsp_flow.py \
#  --what_to_do "imputation_finetune" \
#  \
#  --seq_len ${LEN_WHOLE} \
#  --data_type ${DATA_TYPE} \
#  --feature_size ${FEAT_SIZE} \
#  --one_channel ${ONE_CHANNEL} \
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
#  --max_epochs 500 \
#  --grad_clip_norm 1.0 \
#  --grad_accum_steps 1 \
#  --early_stop "true" \
#  --patience 500 \
#  \
#  --wandb_project ${WANDB_PROJECT} \
#  --wandb_run "impute_finetune_lr${LR}" \
#  \
#  --ckpt_dir ${FINETUNE_CKPT} \
#  --pretrained_ckpt ${PRETRAIN_CKPT} \
#  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
#  \
#  --generated_path "none" \
#  \
#  --gpu_id ${GPU_ID}
#
#
#python dsp_flow.py \
#  --what_to_do "posterior_impute_sample" \
#  \
#  --seq_len ${LEN_WHOLE} \
#  --data_type ${DATA_TYPE} \
#  --feature_size ${FEAT_SIZE} \
#  --one_channel ${ONE_CHANNEL} \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  \
#  --raw_data_paths_train ${DATA_PATHS} \
#  --raw_data_paths_test ${TEST_DATA_PATHS} \
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
#  --gpu_id ${GPU_ID}



#python dsp_flow.py \
#  --what_to_do "anomaly_evaluate" \
#  \
#  --seq_len ${LEN_WHOLE} \
#  --data_type ${DATA_TYPE} \
#  --feature_size ${FEAT_SIZE} \
#  --one_channel ${ONE_CHANNEL} \
#  \
#  --n_layer_enc 4 \
#  --n_layer_dec 4 \
#  --d_model 64 \
#  --n_heads 4 \
#  \
#  --raw_data_paths_train ${DATA_PATHS} \
#  --raw_data_paths_test ${TEST_DATA_PATHS} \
#  --indices_paths_train "[]" \
#  --indices_paths_test ${FINETUNE_TEST_INDICES_PATHS} \
#  --indices_paths_anomaly_for_sample "[]" \
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
#  --ckpt_dir "${FINETUNE_CKPT}" \
#  --pretrained_ckpt "none" \
#  --vqvae_ckpt "${VQVAE_CKPT}/vqvae.pt" \
#  \
#  --generated_path "${FINETUNE_CKPT}/posterior_impute_samples.pth" \
#  \
#  --gpu_id ${GPU_ID}


cd ./dsp_our_method
