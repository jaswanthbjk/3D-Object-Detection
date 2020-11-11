#!/bin/bash
#SBATCH --job-name=frustum-Pointnet
#SBATCH --partition=any
#SBATCH --mem=64G            # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --time=0-72:00:00           # HH-MM-SS
#SBATCH --output=/home/jbandl2s/train.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error=/home/jbandl2s/train.%j.err  # filename for STDERR
#SBATCH --gres=gpu:0


# load cuda
module load cuda

# activate environment
#source ~/anaconda3/bin/activate ~/anaconda3/envs/RnD

# locate to your root directory 
cd /home/jbandl2s/RnD/frustum-pointnets

# run the script
# --gen_train --gen_val --gen_test --gen_val_rgb_detection
python kitti/prepare_data.py --gen_val_rgb_detection
