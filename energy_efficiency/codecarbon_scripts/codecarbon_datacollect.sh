#!/bin/bash

cd .. # go to parent dir to run

# This script is used to automate the collection of codecarbon data for different models and configurations.

# WHAT IT DOES:
# 1. automate codecarbon data collection for:
#     - switch-transformer - simple
#     - qwen-moe - simple
# 2. loop with different parameters:
#     - number of GPUs
#     - batch size
#     - number of experts
#     - dataset split
#     - run number
#     - learning rate


# USAGE: 
# ./codecarbon_datacollect.sh
# TODO (greta) NOTE:
# if you want to run with different parameters, simply change the parameters for switch-transformer and for qwen-moe
# to run with different trainers, comment/uncomment the lines in the run_model functions. or simply comment/uncomment the lines in the loop for the simple trainer

# ------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------ #

################################################################################
##########################    SWITCH TRANSFORMERS    ###########################
################################################################################

# function that runs the model with the given parameters - SWITCH TRANSFORMERS
# input parameters: $1 number of GPUs, $2 batch size, $3 number of experts, $4 data set split, $5 run_num, $6 learning rate
#NOTE: (running simple with less experts)
run_model_switch_simple() {
    local num_gpus=$1
    local batch_size=$2
    local num_experts=$3
    local dataset_split=$4
    local run_num=$5
    local learn_rate=$6

    # Trainers: simple
    num_experts=$((num_experts / num_gpus))

    echo "Running switch-transformer models with $num_gpus GPUs, batch size $batch_size, $num_experts experts, dataset split '$dataset_split', run number $run_num, learning rate $learn_rate"
    echo "$run_num,$num_gpus,$batch_size,$num_experts,$dataset_split,$learn_rate,simple" >> codecarbonlogs/run_log_switch.csv
    
    echo "----------------------------------------------------"
    echo "RUNNING SWITCH-TRANSFORMER SIMPLE"
    echo "----------------------------------------------------"
    # switch-transformer - simple
    python3 launch.py \
        --model switch-base-n \
        --trainer simple \
        --batch_size "$batch_size" \
        --dataset_split "$dataset_split" \
        --switch_transformer_num_experts "$num_experts" \
        --learning_rate "$learn_rate" \
        --train_stats codecarbon \
        --project_name "switch-$num_experts-simple" \
        --run_num "$run_num" 
}

################################################################################
#################################    QWEN    ###################################
################################################################################
# function that runs the model with the given parameters - QWEN-MOE
# input parameters: $1 number of GPUs, $2 batch size, $3 number of experts, $4 data set split, $5 run_num, $6 learning rate
#NOTE: (running simple with less experts)
run_model_qwen_simple() {
    local num_gpus=$1
    local batch_size=$2
    local num_experts=$3
    local dataset_split=$4
    local run_num=$5
    local learn_rate=$6

    # Trainers: simple - divide the number of experts by the number of GPUs

    num_experts=$((num_experts / num_gpus))

    echo "Running qwen-moe models with $num_gpus GPUs, batch size $batch_size, $num_experts experts, dataset split '$dataset_split', run number $run_num, learning rate $learn_rate"
    echo "$run_num,$num_gpus,$batch_size,$num_experts,$dataset_split,$learn_rate,simple" >> codecarbonlogs/run_log_qwen.csv
    echo "----------------------------------------------------"

    mkdir -p codecarbonlogs/qwen-moe-$num_experts-simple

    echo "RUNNING QWEN-MOE SIMPLE"
    echo "----------------------------------------------------"
    # qwen moe - simple
    python3 launch.py \
        --model qwen-moe \
        --trainer simple \
        --batch_size "$batch_size" \
        --dataset_split "$dataset_split" \
        --qwen_num_experts "$num_experts" \
        --learning_rate "$learn_rate" \
        --train_stats codecarbon \
        --project_name "qwen-moe-$num_experts-simple" \
        --run_num "$run_num" "$@"
}

# ------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------ #


# TODO (greta): NOTE: the number of experts is hardcoded in the directory name - remember to change it if you are modifying the number og experts

# create the directories for the codecarbon data files. 
mkdir -p codecarbonlogs
mkdir -p codecarbonlogs/switch-32-simple # NOTE (running simple with less experts) 
mkdir -p codecarbonlogs/switch-16-simple # NOTE (running simple with less experts)
mkdir -p codecarbonlogs/qwen-moe-64-simple # NOTE (running simple with less experts)

# create the directories for the codecarbon data files for plots
mkdir -p codecarbonlogs/plots

# create the directories for the codecarbon data files for losses
mkdir -p codecarbonlogs/losses
mkdir -p codecarbonlogs/losses/switch-32-simple
mkdir -p codecarbonlogs/losses/switch-16-simple
mkdir -p codecarbonlogs/losses/qwen-moe-64-simple

# save the different run parameters in a file
echo "run_num,num_gpus,batch_size,num_experts,dataset_split,learn_rate,type" > codecarbonlogs/run_log_switch.csv
echo "run_num,num_gpus,batch_size,num_experts,dataset_split,learn_rate,type" > codecarbonlogs/run_log_qwen.csv


# ------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------ #

################################################################################
##########################    SWITCH TRANSFORMERS    ###########################
################################################################################

# --- SWITCH-TRANSFORMERS PARAMETERS ---
run_num=1
batch_sizes=(2 8)
num_gpus_values=(8)
num_experts_values=(128)
num_iterations=(10000)
learn_rates=("1e-7")
# run the models with the given parameters
for num_gpus in "${num_gpus_values[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do 
        for num_experts in "${num_experts_values[@]}"; do
        for iterations in "${num_iterations[@]}"; do
            dataset_split="train[:$((num_gpus * batch_size * iterations))]"
			dataset_split_simple="train[:$((batch_size * iterations))]"
                for learn_rate in "${learn_rates[@]}"; do
                    echo "----------------------------------------"
                    echo "Run $run_num config:"
                    echo "  GPUs        : $num_gpus"
                    echo "  Batch size  : $batch_size"
                    echo "  Num experts : $num_experts"
                    echo "  Dataset     : $dataset_split"
                    echo "----------------------------------------"
                    # run the simple trainer
                    run_model_switch_simple "$num_gpus" "$batch_size" "$num_experts" "$dataset_split_simple" "$run_num" "$learn_rate" 
                    run_num=$((run_num + 1))
                done
            done
        done
    done
done


################################################################################
#################################    QWEN    ###################################
################################################################################

# --- QWEN-MOE PARAMETERS ---
run_num=1
batch_sizes=(2 4 6)
num_gpus_values=(8)
num_experts_values=(64)
num_iterations=(10000)
learn_rates=("1e-6")
# run the models with the given parameters
for num_gpus in "${num_gpus_values[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do 
        for num_experts in "${num_experts_values[@]}"; do
        for iterations in "${num_iterations[@]}"; do
            dataset_split="train[:$((num_gpus * batch_size * iterations))]"
            dataset_split_simple="train[:$((batch_size * iterations))]"
                for learn_rate in "${learn_rates[@]}"; do			
                    echo "----------------------------------------"
                    echo "Run $run_num config:"
                    echo "  GPUs        : $num_gpus"
                    echo "  Batch size  : $batch_size"
                    echo "  Num experts : $num_experts"
                    echo "  Dataset     : $dataset_split"
                    echo "----------------------------------------"
                    # run the simple trainer
                    run_model_qwen_simple "$num_gpus" "$batch_size" "$num_experts" "$dataset_split_simple" "$run_num" "$learn_rate" 
                    run_num=$((run_num + 1))
                done
            done
        done
    done
done
