import os, json
import time
import numpy as np


def write_jobscript():
    with open(filename, "w") as f:
        f.write(
            f"""#!/usr/bin/env zsh

#SBATCH --account rwth0934
#SBATCH --job-name sample_{tag}

#SBATCH --output /home/bn227573/out/sample_{tag}_%J.log
#SBATCH --error /home/bn227573/out/sample_{tag}_%J_err.log

#SBATCH --time 150

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 2G

#SBATCH --gres=gpu:1

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

cd /home/bn227573/
conda activate torchProd
cd {REPO_DIR}

python sample_jets.py \\
    --model_dir {model_dir} \\
    --model_name {model_name} \\
    --savetag {savetag} \\
    --num_samples {num_samples} \\
    --batchsize {batchsize} \\
    --num_const {num_const} \\
    --seed {seed}
"""
        )

REPO_DIR = "/home/bn227573/Projects/Transformers/final_repo"

mother_dir = f"{REPO_DIR}/models/top_final/"
tmp = os.listdir(mother_dir)
MODEL_DIRS = [os.path.join(mother_dir, x) for x in ["top_hl8_hd256_3", "top_hl8_hd256_4"]]
MODEL_NAMES = ["model_last.pt"] * len(MODEL_DIRS)
SAVETAGS = ["train_100"] * len(MODEL_DIRS)
NUM_SAMPLES = [200000] * len(MODEL_DIRS)
NUM_CONST = [100] * len(MODEL_DIRS)
SEEDS = [int(time.time())] * len(MODEL_DIRS)

n = 0
for i in range(len(MODEL_DIRS)):
    n += 1
    filename = f"jobscript_{n}.sh"
    print(MODEL_DIRS[i])
    model_dir = MODEL_DIRS[i]
    model_name = MODEL_NAMES[i]
    savetag = SAVETAGS[i]
    num_samples = NUM_SAMPLES[i]
    batchsize = 100
    num_const = NUM_CONST[i]
    seed = SEEDS[i]
    tag = model_dir.split("/")[-1]
    write_jobscript()

    os.system(f"sbatch {filename}")
    time.sleep(1)
