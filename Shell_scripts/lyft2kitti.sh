#!/bin/bash
#SBATCH --job-name=lyft2kitti
#SBATCH --partition=any
#SBATCH --mem=64G            # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --time=0-72:00:00           # HH-MM-SS
#SBATCH --output=/home/jbandl2s/train.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error=/home/jbandl2s/train.%j.err  # filename for STDERR
#SBATCH --gres=gpu:0


# load cuda
# module load cuda
# activate environment
#source ~/anaconda3/bin/activate ~/anaconda3/envs/RnD

# locate to your root directory 
python -m lyft_dataset_sdk.utils.export_kitti nuscenes_gt_to_kitti \
        --lyft_dataroot '/scratch/jbandl2s/v1.02-train' \
        --table_folder '/scratch/jbandl2s/v1.02-train/v1.02-train' \
        --get_all_detections False \
        --store_dir '/scratch/jbandl2s/v1.02-train/lyft_kitti/'

# run the script
