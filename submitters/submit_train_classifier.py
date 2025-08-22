import os, json
import time
import numpy as np


def write_jobscript():
    with open("jobscript.sh", "w") as f:
        f.write(
            f"""#!/usr/bin/env zsh

#SBATCH --job-name {tag}

#SBATCH --output /home/bn227573/out/{tag}_%J.log
#SBATCH --error /home/bn227573/out/{tag}_%J_err.log

#SBATCH --time 360

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 4G

#SBATCH --gres=gpu:1

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

cd /home/bn227573/
conda activate torchProd
cd Projects/Transformers/physics_transformers

python train_classifier.py --bg {bg} \\
    --sig {sig} \\
    --num_events {N} \\
    --num_const {c} \\
    --num_epochs {E} \\
    --log_dir {savedir} \\
    --lr 0.0001
"""
        )


data_files = {}
data_files["qcd"] = "inputs/top_benchmark/discretized/train_qcd_pt40_eta30_phi30_lower001.h5"
data_files["top"] = "inputs/top_benchmark/discretized/train_top_pt40_eta30_phi30_lower001.h5"


bgs = [data_files["qcd"]]
sigs = [data_files["top"]]
tags = [
    "topVSqcd",
]


print(bgs, sigs)
for x, y, z in zip(bgs, sigs, tags):
    bg = x
    sig = y
    N = 600000
    E = 50
    c = 100
    savedir = f"models/class/{z}"
    tag = z
    write_jobscript()
    # os.system("sbatch jobscript.sh")
    time.sleep(1)
