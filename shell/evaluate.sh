CUDA_ID=$1
output_dir=$2
model_path=$(ls -d $output_dir*)
# base_model="/home/ubuntu/CoLLM/llama-7b"
base_model='baffo32/decapoda-research-llama-7B-hf'
test_data="./data/book/sample_test.json"
for path in $model_path
do
    echo $path
    CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
        --base_model $base_model \
        --lora_weights $path \
        --test_data_path $test_data \
        --result_json_data $2.json
done
