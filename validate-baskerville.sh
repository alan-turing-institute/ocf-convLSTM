#!/bin/bash
#SBATCH --qos turing
#SBATCH --account=vjgo8416-climate
#SBATCH --nodes 1
#SBATCH --gpus-per-task 1
#SBATCH --tasks-per-node 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --time 45:00:00
#SBATCH --job-name ConvLSTM-validation

# drop into baskerville
module purge
module restore system
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0
cd /bask/projects/v/vjgo8416-climate/shared/cloudcasting-validation

# set wandb credentials
export WANDB_API_KEY=48ca1d1b17503757e000a50299cb77d5cc548a9a

# check if repo exists
if [ ! -d "ocf-convLSTM" ]; then
    echo "Repo does not exist; cloning..."
    git clone https://github.com/alan-turing-institute/ocf-convLSTM.git
fi
cd ocf-convLSTM

# ensure that we have the latest changes
git pull

# create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Virtual environment does not exist; creating..."
    python -m venv .venv
fi
source .venv/bin/activate
pip install -e .

# upgrade jax to run metrics on GPU
pip install --upgrade "jax[cuda12]"

# run validation
cloudcasting validate
