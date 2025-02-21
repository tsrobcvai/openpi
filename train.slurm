#!/bin/bash
#SBATCH --job-name=Pi0      # Job name
#SBATCH --output=/n/fs/rebar/openpi/slurm_log/Pi0__%A_%a.out  # Output file
#SBATCH --error=/n/fs/rebar/openpi/slurm_log/Pi0__%A_%a.err   # Error file
#
#SBATCH --account=seas
#SBATCH --array=0-1                     # Array of jobs
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --ntasks=1
#
# Request 1 task
#SBATCH --cpus-per-task=8              # Request x CPU cores per task
#SBATCH --mem=48G                       # Request x GB of memory per task
#SBATCH --gres=gpu:1                    # Request 1 GPUs per task
#SBATCH --time=48:00:00                 # Maximum time for each task

# Define the commands to run
. /etc/profile

module purge
module load anaconda3/2024.02

source /u/ts2491/.bashrc
conda activate /n/fs/rebar/openpi/conda_env/pi0
# Calculate the -i parameter based on the job array ID
i_value=$(($SLURM_ARRAY_TASK_ID))
export TORCH_HOME=/n/fs/rebar/.cache/torch
export HF_DATASETS_CACHE=/n/fs/rebar/.cache/huggingface/training

# Run the appropriate training script based on the i_value
if [ $i_value -eq 0 ]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi0_act_rebar_low_mem_finetune \
    --exp-name=pi0_lora_insert_small \
    --overwrite
elif [ $i_value -eq 1 ]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi0_act_rebar_low_mem_finetune_relative \
    --exp-name=pi0_lora_insert_small_relative \
    --overwrite

else
    echo "Invalid i_value: $i_value"
    exit 1
fi
