#!/usr/bin/env python3

"""
Load a checkpoint saved by train_ublund.py, generate Lund-sequence samples,
and make plots using the plotting utility already defined in train_ublund.py.

This script intentionally reuses existing functions/classes from train_ublund.py
instead of redefining them.
"""

import os
import argparse
import torch
from torchinfo import summary

# Reuse existing utilities and model definitions from the training script
from train_ublund import (
    get_loaders,
    quickLundPlot,
    test_model,
    test_modelMDN,
    test_modelCNF,
)
from helpers_train import set_seeds


def parse_args():
    """CLI arguments for checkpoint-based generation."""
    p = argparse.ArgumentParser(description="Load a trained model, generate samples, and plot results")

    # Checkpoint / output
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint file saved by train_ublund.py, e.g. models/test/checkpoints/best.pt",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for plots. Default: checkpoint parent log directory",
    )

    # Data
    p.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Optional override for training file path. If omitted, use the checkpoint args value.",
    )
    p.add_argument(
        "--val-file",
        type=str,
        default=None,
        help="Optional override for validation file path. If omitted, use the checkpoint args value.",
    )
    p.add_argument("--batch-size", type=int, default=None, help="Optional override for batch size")
    p.add_argument("--num-workers", type=int, default=None, help="Optional override for number of workers")
    p.add_argument("--shuffle", action="store_true", default=False, help="Shuffle the loader")

    # Runtime
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cpu", "cuda"],
        help="Force device; default auto",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=-1,
        help="Number of batches to use for generation/plotting. -1 means all batches.",
    )
    p.add_argument(
        "--use-val",
        action="store_true",
        default=False,
        help="Use validation loader instead of training loader for generation",
    )
    p.add_argument(
        "--hist2d-xrange",
        type=float,
        nargs=2,
        default=[-3.0, 7.0],
        help="2D histogram x-range: xmin xmax",
    )
    p.add_argument(
        "--hist2d-yrange",
        type=float,
        nargs=2,
        default=[-5.0, 7.0],
        help="2D histogram y-range: ymin ymax",
    )
    p.add_argument(
        "--hist1d-ranges",
        type=float,
        nargs="+",
        default=None,
        help="Flattened 1D ranges, e.g. --hist1d-ranges -3 7 -5 7",
    )
    return p.parse_args()


def build_model_from_checkpoint(checkpoint, sample_batch, device):
    """
    Rebuild the correct model architecture from checkpoint metadata
    and load the saved state_dict.
    """
    ckpt_args = checkpoint.get("args", {})
    do_cnf = checkpoint.get("doCNF", ckpt_args.get("cnf", False))
    do_mdn = checkpoint.get("doMDN", ckpt_args.get("mdn", True))

    input_dim = sample_batch.shape[2]
    embed_dim = ckpt_args.get("embed_dim", 256)
    num_heads = ckpt_args.get("num_heads", 1)
    num_layers = ckpt_args.get("num_layers", 2)
    ff_dim = ckpt_args.get("ff_dim", 128)
    n_mix = ckpt_args.get("n_mix", 25)
    flow_hidden = ckpt_args.get("flow_hidden", 128)

    if do_cnf:
        model = test_modelCNF(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            flow_hidden=flow_hidden,
        )
    elif do_mdn:
        model = test_modelMDN(
            input_dim=input_dim,
            n_mix=n_mix,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
        )
    else:
        model = test_model(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, do_cnf, do_mdn


def resolve_data_args(cli_args, checkpoint):
    """Resolve data-loading arguments, preferring CLI overrides over checkpoint args."""
    ckpt_args = checkpoint.get("args", {})

    train_file = cli_args.train_file or ckpt_args.get("train_file", "inputFiles/discretized/qcd_lund_cut_train.h5")
    val_file = cli_args.val_file or ckpt_args.get("val_file", "inputFiles/discretized/qcd_lund_cut_val.h5")
    batch_size = cli_args.batch_size or ckpt_args.get("batch_size", 256)
    num_workers = cli_args.num_workers if cli_args.num_workers is not None else ckpt_args.get("num_workers", 1)

    return train_file, val_file, batch_size, num_workers


def default_out_dir(checkpoint_path, cli_out_dir):
    """
    If --out-dir is not given, write to the checkpoint log directory.
    For example: models/test/checkpoints/best.pt -> models/test
    """
    if cli_out_dir:
        return cli_out_dir
    return os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))


def main():
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})

    epoch_losses = checkpoint.get("train_epoch_losses", None)
    loss_curves = checkpoint.get("loss_curves", None)

    seed = ckpt_args.get("seed", 0)
    set_seeds(seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device
    print(f"Running on device: {device}")

    train_file, val_file, batch_size, num_workers = resolve_data_args(args, checkpoint)
    out_dir = default_out_dir(args.checkpoint, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from:\n  train = {train_file}\n  val   = {val_file}")

    train_loader, val_loader = get_loaders(
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=args.shuffle,
    )

    loader = val_loader if args.use_val else train_loader
    first_batch = next(iter(loader))
    print(f"Input shape: {first_batch.shape}")

    model, do_cnf, do_mdn = build_model_from_checkpoint(checkpoint, first_batch, device)

    # Move the summary input to the same device as the model
    summary_input = first_batch[:, :-1, :].to(device)

    summary(
        model,
        input_data=[summary_input],
        col_names=["input_size", "output_size", "num_params", "params_percent", "mult_adds", "trainable"],
    )

    with torch.no_grad():
        original = torch.empty([0, first_batch.shape[-1]], device="cpu")
        generated = torch.empty([0, first_batch.shape[-1]], device="cpu")

        for batch_idx, X in enumerate(loader):
            if args.max_batches >= 0 and batch_idx >= args.max_batches:
                break

            X = X.to(device)
            start = X[:, 0, :].unsqueeze(1)

            generated_seq = model.generate(start, steps=X.shape[1] - 1)

            if batch_idx == 0:
                print("Input example:")
                print(X[0].detach().cpu())
                print("Generated example:")
                print(generated_seq[0].detach().cpu())

            original = torch.cat([original, X.detach().cpu().flatten(0, 1)], dim=0)
            generated = torch.cat([generated, generated_seq.detach().cpu().flatten(0, 1)], dim=0)

    hist2d_range = [args.hist2d_xrange, args.hist2d_yrange]

    hist1d_ranges = None
    if args.hist1d_ranges is not None:
        if len(args.hist1d_ranges) % 2 != 0:
            raise ValueError("--hist1d-ranges must contain an even number of values")
        hist1d_ranges = []
        for i in range(0, len(args.hist1d_ranges), 2):
            hist1d_ranges.append([args.hist1d_ranges[i], args.hist1d_ranges[i + 1]])

    # Reuse the existing plotting utility from train_ublund.py
    quickLundPlot(
        [original.numpy(), generated.numpy()],
        labels=["original", "generated"],
        out_dir=out_dir,
        epoch_losses=epoch_losses,
        loss_curves=loss_curves,
        hist2d_range=hist2d_range,
        hist1d_ranges=hist1d_ranges,
    )

    print(f"Plots written to: {out_dir}")
    print(f"Loaded checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint main loss: {checkpoint.get('loss', 'unknown')}")
    print(f"Model mode: {'CNF' if do_cnf else ('MDN' if do_mdn else 'Regression')}")


if __name__ == "__main__":
    main()
