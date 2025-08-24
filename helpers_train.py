import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from preprocess import preprocess_dataframe


def get_lin_scheduler(num_epochs, num_batches, lr_decay, optimizer):
    training_steps = num_epochs * num_batches
    lr_fn = lambda step: 1.0 - ((1.0 - lr_decay) * (step / training_steps))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_fn)
    return scheduler


def get_exp_scheduler(
    num_epochs: int,
    num_batches: int,
    optimizer: torch.optim.Optimizer,
    final_reduction=1e-2,
):
    training_steps = num_epochs * num_batches
    lr_fn = lambda step: final_reduction ** (step / training_steps)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_fn)
    return scheduler


def get_cos_scheduler(num_epochs, num_batches, optimizer, eta_min=1e-6):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        # T_0=len(train_loader)//5,
        T_0=num_batches * num_epochs + 1,
        eta_min=eta_min,
    )
    return scheduler


def save_opt_states(optimizer, scheduler, scaler, log_dir):
    torch.save(
        {
            "opt_state_dict": optimizer.state_dict(),
            "sched_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        os.path.join(log_dir, "opt_state_dict.pt"),
    )


def load_opt_states(optimizer, scheduler, scaler, log_dir):
    state_dicts = torch.load(os.path.join(log_dir, "opt_state_dict.pt"))
    optimizer.load_state_dict(state_dicts["opt_state_dict"])
    scheduler.load_state_dict(state_dicts["sched_state_dict"])
    scaler.load_state_dict(state_dicts["scaler_state_dict"])


def save_model(model, log_dir, name):
    torch.save(model, os.path.join(log_dir, f"model_{name}.pt"))


def load_model(model_path):
    model = torch.load(model_path)
    print(f"Model uses tanh {model.tanh}")
    return model


def parse_input():
    parser = ArgumentParser()
    parser.add_argument(
        "--log_dir", type=str, default="models/test", help="Model directory"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="inputFiles/top_benchmark/train_qcd_30_bins.h5",
        help="Path to training data file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model file to load",
    )
    parser.add_argument(
        "--sample_file",
        type=str,
        default=None,
        help="Path to file for sampling. If none given, sample from model",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="the random seed for torch and numpy"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Training steps between logging"
    )
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        default=5000,
        help="Training steps between saving checkpoints",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    parser.add_argument(
        "--num_const", type=int, default=100, help="Number of constituents"
    )
    parser.add_argument(
        "--limit_const",
        action="store_true",
        help="Only use jets with at least num_const constituents",
    )
    parser.add_argument(
        "--num_events", type=int, default=10000, help="Number of events for training"
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        nargs=3,
        default=[41, 31, 31],
        help="Number of bins per feature",
    )
    parser.add_argument(
        "--start_token",
        action="store_true",
        help="Whether to use a start particle (learn first particle as well)",
    )
    parser.add_argument(
        "--end_token",
        action="store_true",
        help="Whether to use a end particle (learn jet length as well)",
    )
    parser.add_argument(
        "--contin", action="store_true", help="Whether to continue training"
    )
    parser.add_argument(
        "--global_step", type=int, default=0, help="Starting point of step counter"
    )
    parser.add_argument(
        "--reverse", action="store_true", help="Whether to reverse pt order"
    )

    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=0.01, help="learning rate decay (linear)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.00001, help="weight decay"
    )

    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dim of the model"
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument(
        "--output",
        type=str,
        default="linear",
        choices=["linear", "embprod"],
        help="Output function",
    )
    parser.add_argument(
        "--tanh", action="store_true", help="Apply tanh as final activation"
    )
    args = parser.parse_args()
    return args


def save_arguments(args):
    tmp = args.log_dir
    i = 0
    while os.path.isdir(tmp):
        i += 1
        tmp = args.log_dir + f"_{i}"

    args.log_dir = tmp
    os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, "arguments.txt"), "w") as f:
        arg_dict = vars(args)
        for k, v in arg_dict.items():
            f.write(f"{k:20s} {v}\n")
    return args


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_data(
    path,
    n_events,
    num_features=3,
    num_bins=(41, 31, 31),
    num_const=20,
    reverse=False,
    start_token=False,
    end_token=False,
    limit_const=True,
    batch_size=100,
    num_workers=4,
    shuffle=True,
):
    df = pd.read_hdf(path, "discretized", stop=n_events)
    x, padding_mask, bins = preprocess_dataframe(
        df,
        num_features=num_features,
        num_bins=num_bins,
        num_const=num_const,
        to_tensor=True,
        reverse=reverse,
        start=start_token,
        end=end_token,
        limit_nconst=limit_const,
    )

    train_dataset = TensorDataset(x, padding_mask, bins)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    return train_loader


if __name__ == "__main__":
    pass
