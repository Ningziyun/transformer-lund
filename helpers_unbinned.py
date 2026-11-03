import argparse
import torch

import models_generative

import helpers

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------
def build_unbinned_model(args_or_dict, input_shape, cond_shape):
    """
    New model from the run arguments (argparse namespace or dict). The model stores its constructor
    arguments in model.config, which save_checkpoint writes out, so restarting/validating only needs
    load_unbinned_model / helpers.load_checkpoint_model.
    """
    input_format = helpers.config_get(args_or_dict, "input_format")
    preprocess = helpers.config_get(args_or_dict, "preprocess")
    architecture = helpers.config_get(args_or_dict, "architecture")

    pad_value=-1
    if input_format=="4vec":
        if preprocess=="standardize" or preprocess=="log" or preprocess=="logstd":
            pad_value=-10
    elif input_format=="relvec":
        if preprocess=="standardize" or preprocess=="logstd":
            pad_value=-10

    if architecture=="MDN":
        max_value=10
        if input_format=="4vec":
            if preprocess is None:
                max_value=3e3
            elif preprocess=="standardize":
                max_value=50
            elif preprocess=="linearize":
                max_value=1
            elif preprocess=="log":
                max_value=11

        return models_generative.model_autoregressive_transformer_MDN(
            input_dim=input_shape[2],
            cond_dim=cond_shape[-1],
            n_mix=helpers.as_int(helpers.config_get(args_or_dict,"MDN_nmix")),
            embed_dim=helpers.as_int(helpers.config_get(args_or_dict,"embed_dim")),
            num_heads=helpers.as_int(helpers.config_get(args_or_dict,"num_heads")),
            num_layers=helpers.as_int(helpers.config_get(args_or_dict,"num_layers")),
            ff_dim=helpers.as_int(helpers.config_get(args_or_dict,"ff_dim")),
            n_register=helpers.as_int(helpers.config_get(args_or_dict,"registers")),
            multi_loss=helpers.as_bool(helpers.config_get(args_or_dict,"multi_loss")),
            max_range=max_value,
            pad_value=pad_value,
        )
    elif architecture=="CNF":
        return models_generative.model_CNF(
            input_dim=input_shape[1]*input_shape[2],
            embed_dim=helpers.as_int(helpers.config_get(args_or_dict,"embed_dim")),
            num_heads=helpers.as_int(helpers.config_get(args_or_dict,"num_heads")),
            num_layers=helpers.as_int(helpers.config_get(args_or_dict,"num_layers")),
            ff_dim=helpers.as_int(helpers.config_get(args_or_dict,"ff_dim")),
            cnf_hidden=helpers.as_int(helpers.config_get(args_or_dict,"cnf_hidden")),
            steps=helpers.as_int(helpers.config_get(args_or_dict,"cnf_steps")),
            time_dim=helpers.as_int(helpers.config_get(args_or_dict,"time_dim")),
        )
    elif architecture=="FM":
        return models_generative.model_FM(
            input_shape=input_shape,
            cond_shape=cond_shape,
            hidden_dim=helpers.as_int(helpers.config_get(args_or_dict,"embed_dim")),
            time_dim=helpers.as_int(helpers.config_get(args_or_dict,"time_dim")),
            Nembed_dim=helpers.as_int(helpers.config_get(args_or_dict,"fm_Nembed_dim")),
            steps=helpers.as_int(helpers.config_get(args_or_dict,"fm_steps")),
            multi_loss=helpers.as_bool(helpers.config_get(args_or_dict,"multi_loss")),
            architecture=helpers.config_get(args_or_dict,"fm_architecture"),
            num_heads=helpers.as_int(helpers.config_get(args_or_dict,"num_heads")),
            num_layers=helpers.as_int(helpers.config_get(args_or_dict,"num_layers")),
            ff_dim=helpers.as_int(helpers.config_get(args_or_dict,"ff_dim")),
            n_register=helpers.as_int(helpers.config_get(args_or_dict,"registers")),
            pad_value=pad_value,
        )
    elif architecture=="NF":
        return models_generative.model_normalizing_flow(
            input_dim=input_shape[1]*input_shape[2],
            num_flows=helpers.as_int(helpers.config_get(args_or_dict,"nflows")),
            latent_dim=helpers.as_int(helpers.config_get(args_or_dict,"embed_dim")),
            flow_type=helpers.config_get(args_or_dict,"nflow_type"),
        )
    elif architecture=="Diffusion":
        return models_generative.model_diffusion(
                input_dim=input_shape[1]*input_shape[2],
                hidden_dim=helpers.as_int(helpers.config_get(args_or_dict,"embed_dim")),
                time_steps=helpers.as_int(helpers.config_get(args_or_dict,"diff_steps")),
                beta_start=helpers.as_float(helpers.config_get(args_or_dict,"diff_beta_start")),
                beta_end=helpers.as_float(helpers.config_get(args_or_dict,"diff_beta_end")),
                mode=helpers.config_get(args_or_dict,"diff_type"),
                time_dim=helpers.as_int(helpers.config_get(args_or_dict,"time_dim")),
                sample_time_steps=helpers.as_int(helpers.config_get(args_or_dict,"diff_steps_sample")),
        )
    elif architecture=="SDE":
        return models_generative.model_score_SDE(
                input_dim=input_shape[1]*input_shape[2],
                hidden_dim=helpers.as_int(helpers.config_get(args_or_dict,"embed_dim")),
                time_dim=helpers.as_int(helpers.config_get(args_or_dict,"time_dim")),
                beta_min=helpers.as_float(helpers.config_get(args_or_dict,"SDE_beta_min")),
                beta_max=helpers.as_float(helpers.config_get(args_or_dict,"SDE_beta_max")),
                mode=helpers.config_get(args_or_dict,"SDE_type"),
        )
    elif architecture=="Transformer":
        return models_generative.model_autoregressive_transformer(
            input_dim=input_shape[2],
            cond_dim=cond_shape[-1],
            embed_dim=helpers.as_int(helpers.config_get(args_or_dict,"embed_dim")),
            num_heads=helpers.as_int(helpers.config_get(args_or_dict,"num_heads")),
            num_layers=helpers.as_int(helpers.config_get(args_or_dict,"num_layers")),
            ff_dim=helpers.as_int(helpers.config_get(args_or_dict,"ff_dim")),
            n_register=helpers.as_int(helpers.config_get(args_or_dict,"registers")),
            multi_loss=helpers.as_bool(helpers.config_get(args_or_dict,"multi_loss")),
        )
    else:
        raise ValueError(f"Unknown unbinned architecture {architecture}")

# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
def parse_input():
    """Parse_Input args. Defaults match the current hard-coded values."""
    parser = argparse.ArgumentParser(description="Train transformer/MDN on Lund data")

    helpers.common_training_args(parser)

    # data
    parser.add_argument("--train-file", help="Path to training .h5")
    parser.add_argument("--test-file", help="Path to testing .h5")

    # Architectures
    parser.add_argument("--architecture","-a",type=str, choices=["Transformer","MDN","NF","CNF","Diffusion","SDE","FM"], help="Architecture to run")
    parser.add_argument("--multi-loss", action="store_true", default=False, help="Use multile objective loss (default: False)")
    parser.add_argument("--conditional", action="store_true", default=False, help="Add conditional information (default: False)")
    parser.add_argument("--registers", type=int, default=0, help="Number of registers for transformers (default:0)")

    # general model parameters
    parser.add_argument("--embed-dim", type=int, default=256, help="Transformer embedding dim")
    parser.add_argument("--time-dim", type=int, default=32, help="Transformer embedding dim")

    # transformer hyperparams (keep defaults = your current test_model defaults)
    parser.add_argument("--num-heads", type=int, default=2, help="Transformer num heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer num layers")
    parser.add_argument("--ff-dim", type=int, default=256, help="Transformer feedforward dim")

    # MDN parameters
    parser.add_argument("--MDN-nmix", type=int, default=10, help="Number of MDN mixtures")

    # CNF paramaters
    parser.add_argument("--cnf-hidden", type=int, default=128, help="CNF vector field hidden size")
    parser.add_argument("--cnf-steps", type=int, default=25, help="CNF Euler steps for integration")

    # FM parameters
    parser.add_argument("--fm-architecture", default="transformer", choices=["mlp","transformer"], help="Type of model for the flow prediction")
    parser.add_argument("--fm-steps", type=int, default=25, help="FM Euler steps for integration")
    parser.add_argument("--fm-Nembed-dim", type=int, default=16, help="Size of mulitplicity embedding dimenson")

    # NF paramaters
    parser.add_argument("--nflows", type=int, default=10, help="NF steps")
    parser.add_argument("--nflow-type", type=str, default="neuralspline_coupling", choices=["realnvp","maf","neuralspline_coupling","neuralspline_autoregressive"], help="Algorithm for the normalizing flows")

    # Diff paramaters
    parser.add_argument("--diff-steps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--diff-beta-start", type=float, default=1e-4, help="Diffusion beta start")
    parser.add_argument("--diff-beta-end", type=float, default=2e-2, help="Diffusion beta end")
    parser.add_argument("--diff-type", type=str, default="DDPM", choices=["DDPM","DDIM"], help="Algorithm for the diffusion")
    parser.add_argument("--diff-steps-sample", type=int, default=1000, help="Diffusion sampleing timesteps")

    # SDE parameters
    parser.add_argument("--SDE-beta-min", type=float, default=0.1, help="SDE beta min")
    parser.add_argument("--SDE-beta-max", type=float, default=20.0, help="SDE beta max")
    parser.add_argument("--SDE-type", type=str, default="sde", choices=["sde","ode"], help="Algorithm for the SDE")

    #Some manipulations
    args=parser.parse_args()
    if args.plot_dir is None:
        args.plot_dir = args.log_dir

    #NOTE preprocessing is applied in the lunddata script now
    if args.input_format=="4vec" or args.input_format=="relvec":
        args.preprocess="logstd"

    return args
