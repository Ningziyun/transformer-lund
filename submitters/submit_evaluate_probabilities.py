import os

def write_file():
    with open(f"jobscripts/jobscript_{n}.sh", "w") as f:
        f.write(
        f"""#!/usr/bin/env zsh
#SBATCH --account rwth0934
#SBATCH --job-name probs_{t}

#SBATCH --output /home/bn227573/out/probs_{t}_%J.log
#SBATCH --error /home/bn227573/out/probs_{t}_%J_err.log

#SBATCH --time 5

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 2500M

#SBATCH --gres=gpu:1

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

cd /home/bn227573/
conda activate torchProd
cd Projects/Transformers/final_repo

python evaluate_probabilities.py \\
    --model {m} \\
    --data {d} \\
    --tag {t} \\
    --num_events {200000} \\
    --num_const 100
    """)


models = [
    "output/Transformers/final_repo/models/qcd_final_1/qcd_hl8_hd256/model_last.pt",
    "output/Transformers/final_repo/models/top_final/top_hl8_hd256/model_last.pt",
    "output/Transformers/final_repo/models/top_final/top_hl8_hd256_nc100/model_last.pt",
]

data_paths = [
    "inputFiles/top_benchmark/discretized/test_qcd_pt40_eta30_phi30_lower001.h5",
    "inputFiles/top_benchmark/discretized/train_qcd_pt40_eta30_phi30_lower001.h5",
    "inputFiles/top_benchmark/discretized/test_top_pt40_eta30_phi30_lower001.h5",
    "inputFiles/top_benchmark/discretized/train_top_pt40_eta30_phi30_lower001.h5",
]

tags = [
    "qcd_test",
    "qcd_train",
    "top_test",
    "top_train",
]

n = 1
for m in models:
    for d, t in zip(data_paths, tags):
        write_file()
        os.system(f"sbatch jobscripts/jobscript_{n}.sh")
        n += 1
