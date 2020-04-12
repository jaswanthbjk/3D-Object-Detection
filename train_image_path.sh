#!/bin/bash
#SBATCH --job-name=frustum-Pointnet
#SBATCH --partition=any
#SBATCH --mem=16G            # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --time=0-4:00:00           # HH-MM-SS
#SBATCH --output=/home/jbandl2s/train.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error=/home/jbandl2s/train.%j.err  # filename for STDERR


# load cuda
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/3DOD_Env

# locate to your root directory 
cd /home/jbandl2s/Reference

# run the script
DATA_FILE="/scratch/jbandl2s/v1.02-train/frustum_data"
MODEL_LOG_DIR="./log_v1_test/"
RESTORE_MODEL_PATH="./log_v1_test/model.ckpt"

python generate_image_file_paths.py
