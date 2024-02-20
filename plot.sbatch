#!/bin/sh
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1            # usually set to 1.
#SBATCH --mem=20G
#SBATCH --mail-type=FAIL       # you will receive an email when all jobs finish running.
#SBATCH --chdir=/tudelft.net/staff-bulk/ewi/insy/MMC/shilunzhang/high_order_backbone/

srun --output="plot.out" -u python ~/high_order_backbone/myplot.py --dataset $1 --beta $2 --theta $3 --order $4
