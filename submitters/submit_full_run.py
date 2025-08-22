import os, json, time, sys

def write_transformer(params, transformer_command):
    with open(params["filename"], "w") as f:
        f.write(
            f"""#!/usr/bin/env zsh

#SBATCH --job-name {params["jobname"]}
#SBATCH --account rwth0934

#SBATCH --output {params["out_file"]}_%J.log
#SBATCH --error {params["out_file"]}_%J_err.log

#SBATCH --time {params["runtime"]}

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu {params["mem_req"]}G

#SBATCH --gres=gpu:1

cd $HOME
source .zshrc
conda activate {params["env"]}
cd {params["dir"]}
{transformer_command}
"""
        )


def write_samples(params, dependency):
    with open(params["filename"], "w") as f:
        f.write(
            f"""#!/usr/bin/env zsh

#SBATCH --job-name {params["jobname"]}
#SBATCH --account rwth0934
#SBATCH --dependency afterok:{dependency}

#SBATCH --output {params["out_file"]}_%J.log
#SBATCH --error {params["out_file"]}_%J_err.log

#SBATCH --time {params["runtime"]}

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu {params["mem_req"]}G

#SBATCH --gres=gpu:1

cd $HOME
source .zshrc
conda activate {params["env"]}
cd {params["dir"]}
{params["sampling_command"]}
"""
        )


def write_pnet_train(params, dependency):
    with open(params["filename"], "w") as f:
        f.write(
            f"""#!/usr/bin/env zsh

#SBATCH --job-name {params["jobname"]}
#SBATCH --account rwth0934
#SBATCH --dependency afterok:{dependency}

#SBATCH --output {params["out_file"]}_%J.log
#SBATCH --error {params["out_file"]}_%J_err.log

#SBATCH --time {params["runtime"]}

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu {params["mem_req"]}G

#SBATCH --gres=gpu:1

cd $HOME
source .zshrc
conda activate {params["env"]}
cd {params["dir"]}
{params["pnet_command"]}
"""
        )


def get_transformer_command(params):
    command = f"""python train.py \\
    --num_epochs {params["num_epochs"]} \\
    --data_path {params["data_path"]} \\
    --seed {params["seed"]} \\
    --log_dir {params["log_dir"]} \\
    --batch_size {params["batch_size"]} \\
    --num_events {params["num_events"]} \\
    --num_const {params["num_const"]} \\
    --num_bins {" ".join(str(i) for i in params["num_bins"])} \\
    --logging_steps {params["logging_steps"]} \\
    --checkpoint_steps {params["checkpoint_steps"]} \\
    --lr {params["lr"]} \\
    --num_layers {params["num_layers"]} \\
    --hidden_dim {params["hidden_dim"]} \\
    --num_heads {params["num_heads"]} \\
    --dropout {params["dropout"]} \\
    {'--start_token' if params["start_token"] else ''} \\
    {'--end_token' if  params["end_token"] else ''}"""
    return command


def get_sampling_command(params):
    command = []
    training_files = []
    for i in range(len(params["savetags"])):
        command.append(f"""python sample_jets.py \\
            --model_dir {params["model_dir"]} \\
            --model_name {params["model_name"]} \\
            --savetag {params["savetags"][i]} \\
            --num_samples {params["num_samples"][i]} \\
            --batchsize {params["batchsize"]} \\
            --num_const {params["num_const"][i]} \\
            --seed {params["seed"][i]} \\
            {f'--trunc {params["trunc"][i]}' if params["trunc"][i] else ''}\n""")
        if params["savetags"][i].startswith("train"):
            training_files.append(
                (i, os.path.join(params["model_dir"], f"samples_{params['savetags'][i]}.h5"))
            )
    return command, training_files


def get_particlenet_command(params, sigfile):
    config_path = params["config_loc"]
    params.pop("config_loc")
    params["data"]["sig_file"] = sigfile
    params["data"]["seed"] = int(time.time())
    json.dump(
        params,
        open(config_path, "w"),
        sort_keys=True,
        indent=2
        )
    command = f"python train.py {config_path}"
    return command


config = json.load(open(sys.argv[1], "r"))
config["Sampling"]["model_dir"] = config["Transformer"]["log_dir"]
transformer_command = get_transformer_command(config["Transformer"])
write_transformer(config["Jobscripts"]["Transformer"], transformer_command)

jID_trafo = os.popen(f'sbatch {config["Jobscripts"]["Transformer"]["filename"]}').read().split()[-1]
time.sleep(1)
samplings, files = get_sampling_command(config["Sampling"])
tmp = config["Jobscripts"]["Sampling"]["filename"]
jIDs_sampling = []
for i, sampling in enumerate(samplings):
    config["Jobscripts"]["Sampling"]["sampling_command"] = sampling
    config["Jobscripts"]["Sampling"]["filename"] = "".join(tmp.split(".")[:-1]) + f"_{i}.sh"
    write_samples(config["Jobscripts"]["Sampling"], jID_trafo)

    jIDs_sampling.append(
        os.popen(f'sbatch {config["Jobscripts"]["Sampling"]["filename"]}').read().split()[-1]
    )

tmp = config["ParticleNet"]["config_loc"]
for idx, file in files:
    time.sleep(1)
    config["ParticleNet"]["config_loc"] = os.path.join(tmp, f"config_pnet_{idx}.json")
    command = get_particlenet_command(config["ParticleNet"], file)
    config["Jobscripts"]["ParticleNet"]["filename"] = f"jobscripts/jobscript_pnet_{idx}.sh"
    config["Jobscripts"]["ParticleNet"]["pnet_command"] = command
    write_pnet_train(config["Jobscripts"]["ParticleNet"], jIDs_sampling[idx])

    os.popen(f'sbatch {config["Jobscripts"]["ParticleNet"]["filename"]}')
    print(idx, file)
