import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from argparse import ArgumentParser

from model import JetTransformerClassifier

from tqdm import tqdm
import pandas as pd
import os

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from helpers_train import (
    get_cos_scheduler,
    save_opt_states,
    parse_input,
    save_model,
    save_arguments,
    set_seeds,
    load_model,
)

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_input():
    parser = ArgumentParser()
    parser.add_argument(
        "--log_dir", type=str, default="models/test", help="Model directory"
    )
    parser.add_argument(
        "--bg",
        type=str,
        default="inputFiles/top_benchmark/train_qcd_30_bins.h5",
        help="Path to background data file",
    )
    parser.add_argument(
        "--sig",
        type=str,
        default="inputFiles/top_benchmark/train_top_30_bins.h5",
        help="Path to signal data file",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="the random seed for torch and numpy"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Training steps between logging"
    )

    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    parser.add_argument(
        "--num_const", type=int, default=100, help="Number of constituents"
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

    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
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
    args = parser.parse_args()
    return args


def load_data(file):
    if file.endswith("npz"):
        dat = np.load(file)["jets"][: args.num_events, : args.num_const]
    elif file.endswith("h5"):
        dat = pd.read_hdf(file, key="discretized", stop=args.num_events)
        dat = dat.to_numpy(dtype=np.int64)[:, : args.num_const * 3]
        dat = dat.reshape(dat.shape[0], -1, 3)
    else:
        assert False, "Filetype for bg not supported"
    dat = np.delete(dat, np.where(dat[:, 0, 0] == 0)[0], axis=0)
    dat[dat == -1] = 0
    return dat


def get_dataloader(
    bgf,
    sigf,
):
    bg = load_data(bgf)
    sig = load_data(sigf)

    print(f"Using bg {bg.shape} from {bgf} and sig {sig.shape} from {sigf}")

    dat = np.concatenate((bg, sig), 0)
    lab = np.append(np.zeros(len(bg)), np.ones(len(sig)))
    padding_mask = dat[:, :, 0] != 0

    idx = np.random.permutation(len(dat))
    dat = torch.tensor(dat[idx])
    lab = torch.tensor(lab[idx])
    padding_mask = torch.tensor(padding_mask[idx])

    train_set = TensorDataset(
        dat[: int(0.9 * len(dat))],
        padding_mask[: int(0.9 * len(dat))],
        lab[: int(0.9 * len(dat))],
    )
    val_set = TensorDataset(
        dat[int(0.9 * len(dat)) :],
        padding_mask[int(0.9 * len(dat)) :],
        lab[int(0.9 * len(dat)) :],
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
    )
    return train_loader, val_loader


def plot_rocs(model, val_loader, tag):
    labels = []
    preds = []
    model.eval()
    with torch.no_grad():
        for x, padding_mask, label in tqdm(
            val_loader, total=len(val_loader), desc=f"Validation Epoch {epoch + 1}"
        ):
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            label = label.to(device)

            logits = model(
                x,
                padding_mask,
            )
            preds.append(logits.cpu().numpy())
            labels.append(label.cpu().numpy())

    preds = np.concatenate(preds, 0)
    labels = np.concatenate(labels, 0)
    fpr, tpr, _ = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(tpr, 1.0 / fpr, label=f"AUC {auc}")
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.set_ylim(0.9, 3e3)
    ax.legend()
    fig.savefig(os.path.join(args.log_dir, f"roc_{tag}.png"))

    fig, ax = plt.subplots(constrained_layout=True)
    ax.hist(preds[labels == 0], histtype="step", density=True, bins=30, label="Bg")
    ax.hist(preds[labels == 1], histtype="step", density=True, bins=30, label="Sig")
    ax.legend()
    fig.savefig(os.path.join(args.log_dir, f"preds_{tag}.png"))

    np.savez(os.path.join(args.log_dir, f"preds_{tag}.npz"), preds=preds, labels=labels)


if __name__ == "__main__":
    args = parse_input()
    save_arguments(args)

    set_seeds(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    num_features = 3
    num_bins = tuple(args.num_bins)

    print(f"Using bins: {num_bins}")

    train_loader, val_loader = get_dataloader(args.bg, args.sig)

    # construct model
    model = JetTransformerClassifier(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_features=num_features,
        dropout=args.dropout,
        num_const=args.num_const,
    )
    print(sum([p.numel() for p in model.parameters()]))
    model.to(device)

    # construct optimizer and auto-caster
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_cos_scheduler(
        num_epochs=args.num_epochs,
        num_batches=len(train_loader),
        optimizer=opt,
    )
    scaler = torch.cuda.amp.GradScaler()

    logger = SummaryWriter(args.log_dir)
    global_step = 0
    loss_list = []
    perplexity_list = []
    min_val_loss = np.inf
    for epoch in range(args.num_epochs):
        model.train()

        for x, padding_mask, label in tqdm(
            train_loader, total=len(train_loader), desc=f"Training Epoch {epoch + 1}"
        ):
            opt.zero_grad()
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            label = label.to(device)

            with torch.cuda.amp.autocast():
                logits = model(x, padding_mask)
                loss = model.loss(logits, label.view(-1, 1))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            loss_list.append(loss.cpu().detach().numpy())

            if (global_step + 1) % args.logging_steps == 0:
                logger.add_scalar("Train/Loss", np.mean(loss_list), global_step)
                logger.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
                loss_list = []
                perplexity_list = []

            global_step += 1

        model.eval()
        with torch.no_grad():
            val_loss = []
            val_perplexity = []
            for x, padding_mask, label in tqdm(
                val_loader, total=len(val_loader), desc=f"Validation Epoch {epoch + 1}"
            ):
                x = x.to(device)
                padding_mask = padding_mask.to(device)
                label = label.to(device)

                logits = model(
                    x,
                    padding_mask,
                )
                loss = model.loss(logits, label.view(-1, 1))
                val_loss.append(loss.cpu().detach().numpy())

            val_loss = np.mean(val_loss)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_model(model, args.log_dir, "best")
            logger.add_scalar("Val/Loss", np.mean(val_loss), global_step)

        save_model(model, args.log_dir, "last")
        save_opt_states(
            optimizer=opt, scheduler=scheduler, scaler=scaler, log_dir=args.log_dir
        )

    plot_rocs(model, val_loader, tag="last")
    model = load_model(os.path.join(args.log_dir, "model_best.pt"))
    plot_rocs(model, val_loader, tag="best")
