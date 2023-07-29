export CUDA_VISIBLE_DEVICES=$1
export WANDB_ENTITY=""
export WANDB_PROJECT=""
export NCCL_P2P_DISABLE=0
export PYTHONPATH='../'

model_name_or_path=$2
dataset_name=$3

echo "== Testing =="
echo "GPU No.: $1"
echo "Backbone: $model_name_or_path"
echo "Dataset: $dataset_name"
echo "================"

model_max_length=2048

python3 main.py \
    --stage "test" \
    --model_name_or_path ${model_name_or_path} \
    --dataset_name ${dataset_name} \
    --model_max_length ${model_max_length} \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --dataloader_drop_last False \
    --evaluation_strategy "no" \
    --eval_steps 1000 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --output_dir None
