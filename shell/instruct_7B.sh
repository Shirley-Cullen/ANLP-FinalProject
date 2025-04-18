echo $1, $2
seed=$2
output_dir="/home/ubuntu/TALLRec/output/models_movie_explain_llama"
# base_model="/home/ubuntu/CoLLM/llama-7b"
base_model='baffo32/decapoda-research-llama-7B-hf'
# train_data="/home/ubuntu/TALLRec/data/movie/annotated_train_43.json"
train_data="/home/ubuntu/TALLRec/data/movie/sample_train_annotated_cleaned.json"
val_data="/home/ubuntu/TALLRec/data/movie/valid.json"
instruction_model="alpaca-lora-7B"
for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in 64
        do
                mkdir -p $output_dir
                echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample"
                CUDA_VISIBLE_DEVICES=$1 python -u finetune_rec.py \
                    --base_model $base_model \
                    --train_data_path $train_data \
                    --val_data_path $val_data \
                    --output_dir ${output_dir}_${seed}_${sample} \
                    --batch_size 128 \
                    --micro_batch_size 32 \
                    --num_epochs 100 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint $instruction_model \
                    --sample $sample \
                    --seed $2
        done
    done
done

