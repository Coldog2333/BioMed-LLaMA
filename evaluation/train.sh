export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_ENTITY=""
export WANDB_PROJECT=""
export NCCL_P2P_DISABLE=0
export PYTHONPATH='../'

model_name_or_path=$1
model_name=$2
dataset_name=$3

echo "== Training =="
echo "Backbone: $model_name_or_path"
echo "Model: $model_name"
echo "Dataset: $dataset_name"
echo "================"

model_max_length=2048
# save checkpoint to
output_dir="../checkpoints/${model_name}-${dataset_name}"

torchrun --nproc_per_node=8 --master_port=4136 main.py \
    --stage "train" \
    --model_name_or_path ${model_name_or_path} \
    --dataset_name ${dataset_name} \
    --bf16 True \
    --output_dir ${output_dir} \
    --model_max_length ${model_max_length} \
    --run_name "${model_name}-${model_max_length}-${dataset_name}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_drop_last False \
    --evaluation_strategy "no" \
    --eval_steps 300 \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config "../fsdp_config.json" \
    --tf32 True
