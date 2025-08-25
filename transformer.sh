#!/bin/bash
#SBATCH -n 12
#SBATCH --mem=192G
#SBATCH -t 48:00:00
# Request a GPU partition node and access to 2 GPU
#SBATCH -p gpu --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ning_yan@brown.edu

source transformer_setup.sh
source transformer_run.sh
