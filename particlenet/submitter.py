import os, json
import time
import numpy as np

submit = True

tag = "qcdVStop_full"

def write_jobscript():
    with open(filename, "w") as f:
        f.write(
            f"""#!/usr/bin/env zsh

#SBATCH --job-name pNet_{tag}_{setting}_{job_n}
#SBATCH --account rwth0934

#SBATCH --output /home/bn227573/out/pnet_{tag}_{setting}_%J.log
#SBATCH --error /home/bn227573/out/pnet_{tag}_{setting}_%J_err.log

#SBATCH --time 400

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 2G

#SBATCH --gres=gpu:1


cd /home/bn227573
source .zshrc
conda activate monoJet

cd Projects/Transformers/final_repo/particlenet

python train.py {config_file}
"""
        )

job_n = 0

config = json.load(open("config_default.json", "r"))

config["data"]["bg_file"] = "inputFiles/top_benchmark/discretized/train_qcd_pt40_eta30_phi30_lower001.h5"
config["data"]["bg_key"] = "discretized"
config["data"]["sig_file"] = "inputFiles/top_benchmark/discretized/train_top_pt40_eta30_phi30_lower001.h5"
config["data"]["sig_key"] = "discretized"
config["data"]["n_const"] = 100
config["data"]["n_jets"] = 200000
config["data"]["seed"] = int(time.time())
config["data"]["bg_noise"] = False
config["data"]["sig_noise"] = False

config["graphnet"]["activation"] = "LeakyReLU"
config["graphnet"]["channels"] = [[64, 64, 64], [128, 128, 128], [256, 256, 256]]
config["graphnet"]["classifier"] = [256, 128, 2]
config["graphnet"]["dropout"] = 0.1
config["graphnet"]["k"] = 16
config["graphnet"]["static"] = False

config["logging"]["logfolder"] = "logs/discrete"

config["mask"] = True

config["training"]["batch_size"] = 256
config["training"]["epochs"] = 75
config["training"]["validation_split"] = 0.1
config["training"]["validation_freq"] = 1
config["training"]["verbose"] = 2



# Run top vs QCD as data test for discrete and continuous data
for setting in ["discretized", "raw"]:
    job_n += 1
    filename = f"jobscripts/jobscript_data_{job_n}.sh"

    config["data"]["bg_key"] = setting
    config["data"]["sig_key"] = setting

    config["logging"][
        "logfolder"
    ] = f"logs/data_qcdVStop_{setting}_full"
    config_file = f"configs/config_data_{job_n}.json"

    json.dump(config, open(config_file, "w"), sort_keys=True, indent=2)
    write_jobscript()

    if submit:
        os.system("sbatch {}".format(filename))
    print(f"Submitted {filename}")

# Run for samples
bg_files = ["inputFiles/top_benchmark/discretized/val_qcd_pt40_eta30_phi30_lower001.h5"] * 2 + \
    ["inputFiles/top_benchmark/discretized/val_top_pt40_eta30_phi30_lower001.h5"]* 2

sig_files = [
    "/home/bn227573/Projects/Transformers/final_repo/models/qcd_final_1/qcd_hl8_hd256/samples_train_100.h5",
    "/home/bn227573/Projects/Transformers/final_repo/models/qcd_final_1/qcd_hl8_hd256/samples_train_top5k_100.h5",
    "/home/bn227573/Projects/Transformers/final_repo/models/top_final/top_hl8_hd256/samples_train_100.h5",
    "/home/bn227573/Projects/Transformers/final_repo/models/top_final/top_hl8_hd256/samples_train_top5k_100.h5",
]
tag = "samples"
setting = "samples"
folders = [
    "logs/qcd_final",
    "logs/qcd_final_topK",
    "logs/top_final",
    "logs/top_final_topK",
]

job_n = 0
for bg, sig in zip(bg_files, sig_files):
    job_n += 1
    filename = f"jobscripts/jobscript_samples_{job_n}.sh"
    config["logging"]["logfolder"] = folders[job_n-1]
    config["data"]["bg_file"] = bg
    config["data"]["bg_key"] = "discretized"
    config["data"]["sig_file"] = sig
    config["data"]["sig_key"] = "discretized"
    config["data"]["seed"] = int(time.time()) + np.random.randint(0, 2**10)
    config_file = f"configs/config_samples_{job_n}.json"
    json.dump(config, open(config_file, "w"), sort_keys=True, indent=2)
    write_jobscript()
    if submit:
        os.system("sbatch {}".format(filename))
    time.sleep(1)
    print(f"Submitted {filename}")
