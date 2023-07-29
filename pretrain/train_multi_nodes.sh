export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_ENTITY=""
export WANDB_PROJECT=""

# rdzv_endpoint="localhost:30426"  # 分布式通信主机

# if the data_path is a file, then we will use this file to train
# if the data_path is a directory, then we will use all the files in this directory to train
data_path="/INPUT/pile/pile-pubmed/train"
model_name_or_path=""

model_max_length=2048
# save checkpoint to (If the output_dir has some checkpoints with trainer_state.json,
# the trainer will be resumed from the last checkpoint)
output_dir="checkpoints/biomed-llama-7b-${model_max_length}-node-$1-full-data"

## Hyperparameters
# Be sure to set N-gpu * per_device_train_batch_size * gradient_accumulation_steps == 204.8
# Otherwise you need to change the learning rate
# learning_rate = 3e-5 / 204.8 * (N-gpu * per_device_train_batch_size * gradient_accumulation_steps)
# e.g., 32 x A100, per_device_bs=1, grad_acc_steps=8, then lr=3.75e-5, but we can set as 3e-5
#       40 x A40, per_device_bs=1, grad_acc_steps=8, then lr=9.375e-5, but we can set as 9e-5
master_addr="127.0.0.1"

torchrun \
  --nproc_per_node=4 \
  --nnodes=8 \
  --node_rank=$1 \
  --master_addr=${master_addr} \
  --master_port=9993 \
  train.py \
    --model_name_or_path ${model_name_or_path} \
    --data_path ${data_path} \
    --bf16 True \
    --output_dir ${output_dir} \
    --model_max_length ${model_max_length} \
    --run_name "biomed-llama-7b-${model_max_length}-full-data" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --dataloader_drop_last True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 3 \
    --ddp_timeout 3600 \
    --save_on_each_node True \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.03 \
    --num_cycles 0.39758361765043326 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "_hybrid_shard_zero2 auto_wrap" \
    --fsdp_config "fsdp_config.json" \
    --tf32 True
