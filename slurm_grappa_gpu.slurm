#!/bin/bash
# This script executes HOROVOD based multi GPU training for subcortical segmentation.
# Written by Oren Solomon
# ------------------------------------------------

# Slurm parameters
#SBATCH --partition sim-normal
#SBATCH --gres gpu:1
#SBATCH --job-name GRAPPA
#SBATCH --export ALL
#SBATCH -n 1
#SBATCH -N 1

source activate DBSV10

twix_file=Path to TWIX file.dat
savefolder=name of save folder
kernel_size="5 5"
lam=0.01

# SLURM parameters
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# HOROVOD and python
echo "     Starting GRAPPA GPU      "  
echo "------------------------------"
python main_tester.py --twix_file $twix_file --savefolder $savefolder --lam $lam --kernel_size $kernel_size --gpu true
echo "SLURM job finished"


