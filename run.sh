
export CUDA_VISIBLE_DEVICES=1
python3 src/train_sft.py \
    --model_name_or_path /root/data_baichuan/baichuan-7B \
    --do_train \
    --dataset 52k,92k \
    --finetuning_type lora \
    --output_dir /root/data_baichuan/output \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 4.0 \
    --plot_loss \
    --fp16 \
    --lora_target W_pack \

