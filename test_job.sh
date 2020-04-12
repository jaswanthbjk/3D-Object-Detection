#!/bin/bash
#SBATCH --job-name=frustum-Pointnet
#SBATCH --partition=gpu
#SBATCH --mem=64G            # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --time=0-48:00:00           # HH-MM-SS
#SBATCH --output=/home/jbandl2s/train.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error=/home/jbandl2s/train.%j.err  # filename for STDERR
#SBATCH --gres=gpu:0


# load cuda
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/3DOD_Env

# locate to your root directory 
cd /home/jbandl2s/Reference

# run the script
DATA_FILE="/scratch/jbandl2s//v1.02-train/artifacts/test_frustums"
TRAINED_MODEL_PATH="./log_v1_test/model.ckpt"

python test_v2.py --gpu 0 --model frustum_pointnets_v1 --batch_size 32 --model_path "./log_v1_test/model.ckpt" --data_dir "/scratch/jbandl2s//v1.02-train/artifacts/test_frustums" --file_pattern "scene_0_train.tfrec"
