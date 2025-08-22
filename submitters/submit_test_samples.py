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

#SBATCH --time 20

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 2G

#SBATCH --gres=gpu:1

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

cd /home/bn227573/
conda activate torchProd
cd Projects/Transformers/physics_transformers

python test_samples.py --bg {bg} \\
    --sig {sig} \\
    -N {N} \\
    -c {c} \\
    -E {E} \\
    --save_dir {savedir}
"""
        )


data_files = {}
data_files["qcd"] = "inputFiles/top_benchmark/test_qcd_30_bins.h5"
data_files["top"] = "inputFiles/top_benchmark/test_top_30_bins.h5"


# for x, y in zip(sample_files, params):
for folder in os.listdir("output/Transformers/models/scan2"):
    bg = data_files["qcd"]
    sig = os.path.join(
        "output/Transformers/models/scan2", folder, "samples_100.npz"
    )
    N = 20000
    E = 20
    c = 100
    savedir = f"sample_tests/{folder}"
    tag = folder
    write_jobscript()
    os.system("sbatch jobscript.sh")
    time.sleep(0.5)
