import os, json
import time
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data_path", required=True, type=str)
parser.add_argument("--log_dir", required=True, type=str)
parser.add_argument("--tag", required=True, type=str)
parser.add_argument("--num_bins", nargs=3, required=True)
args = parser.parse_args()

with open("jobscript.sh", "w") as f:
    f.write(
        f"""#!/usr/bin/env zsh
#SBATCH --account rwth0934
#SBATCH --job-name trade_{args.tag}

#SBATCH --output /home/bn227573/out/trade_{args.tag}_%J.log
#SBATCH --error /home/bn227573/out/trade_{args.tag}_%J_err.log

#SBATCH --time 360

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 2500M

#SBATCH --gres=gpu:1

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

cd /home/bn227573/
conda activate torchProd
cd Projects/Transformers/final_repo

python train.py \\
    --num_epochs 50 \\
    --data_path {args.data_path} \\
    --seed {int(time.time())} \\
    --log_dir {args.log_dir} \\
    --batch_size 100 \\
    --num_events 600000 \\
    --num_const 50 \\
    --num_bins {" ".join(args.num_bins)} \\
    --logging_steps 50 \\
    --checkpoint_steps 12000 \\
    --lr 5e-4 \\
    --num_layers 8 \\
    --hidden_dim 256 \\
    --num_heads 4 \\
    --dropout 0.1 \\
    --start_token \\
    --end_token

"""
        )
os.system("sbatch jobscript.sh")
time.sleep(1)

