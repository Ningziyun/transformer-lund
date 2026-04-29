
#module load miniforge root
#source /oscar/runtime/software/external/miniforge/23.11.0-0/etc/profile.d/conda.sh
#. ~/.conda_init
#conda activate torch_env9

. ~/.conda_init
#module load python/3.13.5
#module load python/3.11
#module load cuda cudnn
#source ~/torch26.venv/bin/activate
#conda init
#module load root pandas
#module load root
#conda activate torch_env9

#module purge
module load cuda
module load cudnn
#module load root
. ~/.conda_init
conda activate /oscar/home/nyan9/.conda/envs/torch_env11
