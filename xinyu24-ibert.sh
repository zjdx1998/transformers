#!/bin/bash
#SBATCH --partition=iris # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --cpus-per-task=4 # Request 4 CPUs for this task
#SBATCH --mem=64G
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name=quant # Name the job (for easier monitoring)
#SBATCH --exclude=iris1,iris2,iris3,iris4

# python examples/pytorch/text-classification/run_glue.py          --model_name_or_path kssteven/ibert-roberta-base          --task_name MRPC          --do_eval True          --do_train True          --evaluation_strategy epoch          --max_seq_length 128          --per_device_train_batch_size 32          --save_steps 500          --learning_rate 2e-5          --num_train_epochs 3          --output_dir ./mrpc_full/          --overwrite_output_dir True          --cache_dir /iris/u/xinyu24/cache
# python examples/pytorch/text-classification/run_glue.py          --model_name_or_path ./mrpc_full_why          --task_name MRPC          --do_eval True          --do_train True          --evaluation_strategy epoch          --max_seq_length 128          --per_device_train_batch_size 32          --save_steps 500          --learning_rate 1e-6          --num_train_epochs 3          --output_dir ./mrpc_quant         --overwrite_output_dir True          --cache_dir /iris/u/xinyu24/cache

export output_dir=$1
export methods=$2
export cache_dir=$3

rm -rf $output_dir
echo "Deleted original output directory."
mkdir $output_dir

# rm -rf $cache_dir
# echo "Deleted original cache directory."
# mkdir $cache_dir

count=$(echo "$methods" | awk -F',' '{print NF}')
for ((i=1; i<=$count; i++)); do
    method=$(echo "$methods" | awk -F',' -v x=$i '{print $x}')
    echo -e "Processing $method.\n"

    mkdir -p $output_dir/$method

    python examples/pytorch/text-classification/run_glue.py \
            --cache_dir $cache_dir \
            --model_name_or_path kssteven/ibert-roberta-base \
            --task_name $method \
            --do_eval \
            --do_train \
            --evaluation_strategy epoch \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --save_steps 115 \
            --learning_rate 1e-6 \
            --num_train_epochs 5 \
            --output_dir $output_dir/$method/ibert-wo-quant \
            --overwrite_output_dir True

    latest_checkpoint=$(find "$output_dir/$method" -type d -name "checkpoint*" | grep -Eo 'checkpoint-[0-9]+' | sort -t '-' -k 2 -n | tail -n 1)
    echo -e "Latest checkpoint is ${latest_checkpoint}\n"

    sed -i 's/"quant_mode": false/"quant_mode": true/' $output_dir/$method/ibert-wo-quant/config.json
    # sed -i 's/"quant_mode": false/"quant_mode": true/' $output_dir/ibert-wo-quant/$latest_checkpoint/config.json

    # if rm $output_dir/ibert-wo-quant/$latest_checkpoint/optimizer.pt; then
    #     echo "Delete optimizer.pt"
    # else
    #     echo "optimizer.pt doesn't exist"
    # fi
    # if rm $output_dir/ibert-wo-quant/$latest_checkpoint/scheduler.pt; then
    #     echo "Delete scheduler.pt"
    # else
    #     echo "scheduler.pt doesn't exist"
    # fi
    # if rm $output_dir/ibert-wo-quant/$latest_checkpoint/trainer_state.json; then
    #     echo "Delete trainer_state.json"
    # else
    #     echo "trainer_state.json doesn't exist"
    # fi

    if rm $output_dir/$method/ibert-wo-quant/optimizer.pt; then
        echo "Delete optimizer.pt"
    else
        echo "optimizer.pt doesn't exist"
    fi
    if rm $output_dir/$method/ibert-wo-quant/scheduler.pt; then
        echo "Delete scheduler.pt"
    else
        echo "scheduler.pt doesn't exist"
    fi
    if rm $output_dir/$method/ibert-wo-quant/trainer_state.json; then
        echo "Delete trainer_state.json"
    else
        echo "trainer_state.json doesn't exist"
    fi

    python examples/pytorch/text-classification/run_glue.py \
            --cache_dir $cache_dir \
            --model_name_or_path $output_dir/$method/ibert-wo-quant \
            --task_name $method \
            --do_eval \
            --do_train \
            --evaluation_strategy epoch \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --save_steps 115 \
            --learning_rate 2e-5 \
            --num_train_epochs 5 \
            --output_dir $output_dir/$method/ibert-w-quant \
            --overwrite_output_dir True

    echo -e "Done for iBert\n"

    python examples/pytorch/text-classification/run_glue.py \
            --cache_dir $cache_dir \
            --model_name_or_path roberta-base \
            --task_name $method \
            --do_eval \
            --do_train \
            --evaluation_strategy epoch \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --save_steps 115 \
            --learning_rate 5e-5 \
            --num_train_epochs 5 \
            --output_dir $output_dir/$method/roberta-full

    echo -e "Done for roberta\n"

    echo -e "This is for $method\n"

done