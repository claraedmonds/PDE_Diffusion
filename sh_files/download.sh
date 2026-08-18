#!/bin/sh
#BSUB -J download
#BSUB -o logs/train_full/0_era5_%J.out
#BSUB -e logs/train_full/0_era5_%J.err
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=2G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1440
####BSUB -u claed@dtu.dk
####BSUB -B
####BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment	
# module load scipy/VERSION

nvidia-smi
module load cuda/11.6
 
# activate the virtual environment 
# NOTE: needs to have been built with the same SciPy version above!
source /zhome/e3/b/155491/miniconda3/etc/profile.d/conda.sh
conda activate pde_diff

python src/pde_diff/data/download_data.py