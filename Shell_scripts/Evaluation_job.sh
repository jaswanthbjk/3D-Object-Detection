#!/bin/bash
#SBATCH --job-name=eval-f-pointnet
#SBATCH --partition=any
#SBATCH --mem=64G            # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --time=0-24:00:00           # HH-MM-SS
#SBATCH --output=/home/jbandl2s/train.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error=/home/jbandl2s/train.%j.err  # filename for STDERR


# load cuda
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/3DOD_Env

# locate to your root directory 
cd /home/jbandl2s/Reference

# run the script

python3 test_making_inference.py --inference_file /scratch/jbandl2s/Lyft_dataset/artifacts/inference_results_kag_test.tfrec --pred_file /scratch/jbandl2s/v1.02-train/artifacts/train_val_pred_kag_test.csv --data_name 'test'
