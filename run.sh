
export CUDA_VISIBLE_DEVICES=0


# cp -rf ./data/dataset_info.json /root/workspace_law/data_baichuan
# python3 src/train_sft.py \
#     --model_name_or_path /root/workspace_law/baichuan-7B \
#     --do_train \
#     --dataset 52k,92k \
#     --dataset_dir /root/workspace_law/data_baichuan \
#     --finetuning_type lora \
#     --output_dir /root/workspace_law/data_baichuan/output \
#     --overwrite_cache \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 4.0 \
#     --plot_loss \
#     --fp16 \
#     --lora_target W_pack \
#     --lora_rank 32 \

# python3 src/cli_demo.py \
#     --model_name_or_path /root/workspace_law/baichuan-7B \
#     --checkpoint_dir /root/workspace_law/data_baichuan/output \
#     --lora_target W_pack \

python3 src/web_demo.py \
    --model_name_or_path /root/data_baichuan/baichuan-7B \
    --checkpoint_dir /root/zdy/LLaMA-Efficient-Tuning/output/baichuan_lora \
    --lora_target W_pack \
