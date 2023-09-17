#!/bin/sh
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1            # usually set to 1.
#SBATCH --mem=30G
#SBATCH --mail-type=NONE       # you will receive an email when all jobs finish running.
#SBATCH --chdir=/tudelft.net/staff-bulk/ewi/insy/MMC/shilunzhang/high_order_backbone/

srun -u python ~/high_order_backbone/myplot.py --dataset ht09 --theta -1
