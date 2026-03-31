#!/usr/bin/env python3

"""
Load a checkpoint saved by train_ublund.py, generate Lund-sequence samples,
and make plots using the plotting utility already defined in train_ublund.py.

This script intentionally reuses existing functions/classes from train_ublund.py
instead of redefining them.
"""

import os
import re
import csv
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from torchinfo import summary

# Reuse existing utilities and model definitions from the training script
from train_ublund import (
    get_loaders,
    test_model,
    test_modelMDN,
    quickLundPlot,
    test_modelCNF,
)
from helpers_train import set_seeds

import textwrap



def parse_args():
    """CLI arguments for checkpoint-based generation."""
    p = argparse.ArgumentParser(description="Load a trained model, generate samples, and plot results")

    # Checkpoint / output
    p.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
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
    p.add_argument(
        "--hist1d-bins",
        type=int,
        default=30,
        help="Number of bins for 1D histograms",
    )
    p.add_argument(
        "--hist1d-logy",
        action="store_true",
        default=False,
        help="Use log scale on the y-axis for 1D histograms",
    )
    p.add_argument(
        "--not-save-both-hist1d-scales",
        action="store_true",
        default=False,
        help="Save both linear-scale and log-scale versions of the 1D histograms",
    )
    p.add_argument(
        "--loss-zoom-percentile",
        type=float,
        default=80.0,
        help="Upper percentile used for the zoomed loss plot to suppress very large early values",
    )
    p.add_argument(
        "--loss-zoom-headroom",
        type=float,
        default=0.08,
        help="Extra fractional headroom added above the zoomed loss upper limit",
    )
    return p.parse_args()
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

def parse_arguments_txt(txt_path):
    """
    Parse key-value pairs from arguments.txt.

    This supports both formats currently written by the training script:
    1) save_arguments(args) style lines
    2) appended summary lines such as:
       resolved_model_mode, scheduler, cos_start_epoch, best_epoch, best_loss, etc.
    """
    meta = {}

    if not os.path.exists(txt_path):
        return meta

    with open(txt_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Support "key: value"
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
                continue

            # Support whitespace-aligned "key    value"
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].strip()
                value = " ".join(parts[1:]).strip()
                meta[key] = value

    return meta


def infer_log_dir_from_checkpoint(checkpoint_path):
    """
    Convert models/test/checkpoints/best.pt -> models/test
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))


def build_run_caption(meta):
    """
    Build legend text from arguments.txt metadata.
    Example:
    MDN bs=100 lr=0.005 cos-d minlr=5e-05
    """
    model_mode = meta.get("resolved_model_mode", "Unknown")
    batch_size = meta.get("batch_size", meta.get("batch-size", "?"))
    start_epoch = meta.get("cos_damping_start_epoch", None)
    total_epoch = meta.get("epochs", meta.get("total_epochs", None))
    lr = meta.get("lr", "?")
    scheduler = meta.get("scheduler", "none")
    cos_final_lr = meta.get("cos_final_lr", None)

    label = f"{model_mode} bs={batch_size} lr={lr}"

    # Add total epoch (always show if available)
    if total_epoch is not None:
        label += f" ep={total_epoch}"

    # Add cos-damping info (only if used)
    if scheduler == "cos_damping":
        label += " cos-osc"

        if start_epoch is not None:
            label += f" ep_st={start_epoch}"

        if cos_final_lr is not None:
            label += f" lr_f={cos_final_lr}"

    return label

def _best_balanced_split(words, n_lines):
    """
    Split a list of words into n_lines so that the longest line is as short as possible.
    This prefers visually balanced legend lines over naive fixed-width wrapping.
    """
    if n_lines <= 1 or len(words) <= 1:
        return [" ".join(words)]

    best_lines = [" ".join(words)]
    best_score = float("inf")

    # Two-line split
    if n_lines == 2:
        for i in range(1, len(words)):
            lines = [
                " ".join(words[:i]),
                " ".join(words[i:]),
            ]
            score = max(len(line) for line in lines)
            if score < best_score:
                best_score = score
                best_lines = lines
        return best_lines

    # Three-line split
    if n_lines == 3:
        for i in range(1, len(words) - 1):
            for j in range(i + 1, len(words)):
                lines = [
                    " ".join(words[:i]),
                    " ".join(words[i:j]),
                    " ".join(words[j:]),
                ]
                score = max(len(line) for line in lines)
                if score < best_score:
                    best_score = score
                    best_lines = lines
        return best_lines

    return [" ".join(words)]


def format_legend_labels(labels, target_chars=28, max_lines=3):
    """
    Format legend labels with balanced line breaks.
    Priority:
    1) keep one line if short enough
    2) use balanced 2-line split
    3) use balanced 3-line split
    """
    formatted = []

    for label in labels:
        if len(label) <= target_chars:
            formatted.append(label)
            continue

        words = label.split()

        # Try 2 lines first
        lines_2 = _best_balanced_split(words, 2)
        if max(len(line) for line in lines_2) <= target_chars:
            formatted.append("\n".join(lines_2))
            continue

        # Then try 3 lines
        if max_lines >= 3:
            lines_3 = _best_balanced_split(words, 3)
            formatted.append("\n".join(lines_3))
            continue

        formatted.append(label)

    return formatted

def load_loss_history_csv(csv_path):
    """
    Load loss_history.csv into a dict of lists.
    Returns {} if the file does not exist.
    """
    if not os.path.exists(csv_path):
        return {}

    curves = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for name in fieldnames:
            if name != "epoch":
                curves[name] = []

        for row in reader:
            for name in curves:
                value = row.get(name, "")
                if value is None or value == "":
                    curves[name].append(np.nan)
                else:
                    curves[name].append(float(value))

    return curves


def make_output_dir(cli_out_dir, checkpoint_paths):
    """
    Resolve output directory for multi-model plotting.
    If --out-dir is given, use it.
    Otherwise, use the parent directory of the first checkpoint.
    """
    if cli_out_dir:
        return cli_out_dir
    return infer_log_dir_from_checkpoint(checkpoint_paths[0])


def generate_original_and_generated(model, loader, device, max_batches):
    """
    Generate flattened original and generated arrays for one model.
    """
    first_batch = next(iter(loader))

    with torch.no_grad():
        original = torch.empty([0, first_batch.shape[-1]], device="cpu")
        generated = torch.empty([0, first_batch.shape[-1]], device="cpu")

        for batch_idx, X in enumerate(loader):
            if max_batches >= 0 and batch_idx >= max_batches:
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

    return original.numpy(), generated.numpy()


def plot_combined_1dhist(inputs,labels,out_dir,hist1d_ranges=None,hist1d_bins=20,logy=False,out_name=None,logy_floor_mode="clamped"):
    """
    Plot one shared original plus multiple generated samples on the same 1D figures.
    Automatically choose a sensible y-range for both linear and log-scale plots.
    """
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    Ndim = inputs[0].shape[1]
    Nin = len(inputs)

    if hist1d_ranges is None:
        hist1d_ranges = [[-3, 7] for _ in range(Ndim)]

    fig, axs = plt.subplots(Ndim, 1, figsize=(8.0, 8.0))
    if Ndim == 1:
        axs = [axs]

    axis_titles = ["log(kt)","log(1/deltaR)"]

    for ii in range(Ndim):
        feature_idx = 1 - ii  # Swap feature order so top panel shows kt and bottom panel shows log(1/deltaR)
        # Store histogram heights so we can set y-limits from actual bin contents
        all_hist_counts = []
        positive_hist_counts = []

        for jj in range(Nin):
            values = inputs[jj][:, feature_idx]

            # Compute the histogram first so the y-limits can be data-driven
            hist_counts, bin_edges = np.histogram(
                values,
                bins=hist1d_bins,
                range=hist1d_ranges[ii],
                density=True,
            )

            all_hist_counts.append(hist_counts)
            positive_hist_counts.extend(hist_counts[hist_counts > 0.0])

            axs[ii].hist(
                values,
                bins=hist1d_bins,
                range=hist1d_ranges[ii],
                histtype="step",
                density=True,
                linestyle=linestyles[jj % len(linestyles)],
                label=labels[jj],
            )

        if ii < len(axis_titles):
            axs[ii].set_title(axis_titles[ii])
        else:
            axs[ii].set_title(f"feature_{ii}")

        axs[ii].set_ylabel("Density")
        axs[ii].set_xlabel("value")

        # Determine the maximum histogram height across all inputs
        ymax = 0.0
        for hist_counts in all_hist_counts:
            if hist_counts.size > 0:
                ymax = max(ymax, np.max(hist_counts))
        
        if logy:
            axs[ii].set_yscale("log")

            if len(positive_hist_counts) > 0:
                min_positive = min(positive_hist_counts)
                max_positive = max(positive_hist_counts)

                # Choose the lower bound strategy for the log-y axis.
                # "clamped": keep the current behavior and do not go below 1e-4.
                # "tail": expose the sparse tail by using the true minimum positive bin scale.
                if logy_floor_mode == "tail":
                    y_min = min_positive / 3.0
                else:
                    y_min = max(min_positive / 3.0, 1e-4)

                # Leave some room above the tallest bin for readability.
                y_max = max_positive * 1.5

                # Guard against degenerate cases
                if y_max <= y_min:
                    y_max = y_min * 10.0

                axs[ii].set_ylim(y_min, y_max)
            else:
                # Fallback if everything is empty
                if logy_floor_mode == "tail":
                    axs[ii].set_ylim(1e-10, 1.0)
                else:
                    axs[ii].set_ylim(1e-4, 1.0)

            # Cleaner minor ticks on log axis
            axs[ii].yaxis.set_major_locator(LogLocator(base=10.0))
            axs[ii].yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            axs[ii].yaxis.set_minor_formatter(NullFormatter())

        else:
            # Linear scale: set the top from the actual histogram maximum
            # instead of using a hard-coded 0.5 for every case
            y_max = ymax * 1.15 if ymax > 0 else 1.0
            axs[ii].set_ylim(0.0, y_max)

    # Format legend labels with balanced line breaks first
    legend_labels = format_legend_labels(labels, target_chars=28, max_lines=3)

    # Start from a moderate font size instead of a large one
    legend_fontsize = 10

    legend = axs[0].legend(
        legend_labels,
        loc="best",
        fontsize=legend_fontsize,
        framealpha=0.9,
        borderpad=0.6,
        labelspacing=0.4,
        handlelength=2.0,
    )

    # Draw the canvas before measuring legend size
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer)

    fig_width_px = fig.get_size_inches()[0] * fig.dpi
    fig_height_px = fig.get_size_inches()[1] * fig.dpi

    # Priority order:
    # 1) balanced wrapping
    # 2) smaller font
    # 3) balanced wrapping + smaller font + move to a safer in-figure position

    # If the legend is still too wide/tall, reduce font size step by step
    if bbox.width > 0.42 * fig_width_px or bbox.height > 0.22 * fig_height_px:
        for fs in [9, 8]:
            for txt in legend.get_texts():
                txt.set_fontsize(fs)
            fig.canvas.draw()
            bbox = legend.get_window_extent(renderer=fig.canvas.get_renderer())

            if bbox.width <= 0.42 * fig_width_px and bbox.height <= 0.22 * fig_height_px:
                break

    # If it is still too large, move it to the upper center inside the first panel
    if bbox.width > 0.42 * fig_width_px or bbox.height > 0.22 * fig_height_px:
        legend.remove()
        legend = axs[0].legend(
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            fontsize=8,
            framealpha=0.9,
            borderpad=0.5,
            labelspacing=0.35,
            handlelength=2.0,
        )
        fig.canvas.draw()

    if out_name is None:
        out_name = "projection_combined_logy" if logy else "projection_combined"

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, out_name + ".png"))
    fig.savefig(os.path.join(out_dir, out_name + ".pdf"))
    plt.close(fig)


def plot_combined_losses(run_infos, out_dir):
    """
    Plot loss curves from multiple runs on shared figures.

    For each metric name found across runs, create one overlay plot.
    """
    metric_names = set()
    for info in run_infos:
        for metric_name, curve in info["loss_curves_csv"].items():
            if curve is not None and len(curve) > 0:
                metric_names.add(metric_name)

    for metric_name in sorted(metric_names):
        fig = plt.figure(figsize=(6.0, 4.0))
        used_any = False

        for info in run_infos:
            curve = info["loss_curves_csv"].get(metric_name, None)
            if curve is None or len(curve) == 0:
                continue

            y = np.asarray(curve, dtype=float)
            x = np.arange(1, len(y) + 1)
            mask = ~np.isnan(y)
            if np.any(mask):
                #plt.plot(x[mask], y[mask], marker="o", label=info["caption"])
                # Use semi-transparent lines without markers to reduce overlap clutter
                plt.plot(
                    x[mask],
                    y[mask],
                    linestyle="-",
                    linewidth=2,
                    alpha=0.6,   # transparency helps reveal overlapping curves
                    label=info["caption"]
                )
                used_any = True

        if not used_any:
            plt.close(fig)
            continue

        plt.xlabel("Epoch")
        plt.ylabel("NLL" if "nll" in metric_name.lower() else "Loss")
        plt.title(f"Loss vs Epoch ({metric_name})")
        plt.grid(True)
        plt.legend()
        fig.tight_layout()

        fig.savefig(os.path.join(out_dir, f"loss_combined__{metric_name}.png"))
        fig.savefig(os.path.join(out_dir, f"loss_combined__{metric_name}.pdf"))
        plt.close(fig)
        
def main():
    args = parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device
    print(f"Running on device: {device}")

    out_dir = make_output_dir(args.out_dir, args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    run_infos = []
    original_shared = None
    generated_inputs = []
    generated_labels = []

    hist1d_ranges = None
    if args.hist1d_ranges is not None:
        if len(args.hist1d_ranges) % 2 != 0:
            raise ValueError("--hist1d-ranges must contain an even number of values")
        hist1d_ranges = []
        for i in range(0, len(args.hist1d_ranges), 2):
            hist1d_ranges.append([args.hist1d_ranges[i], args.hist1d_ranges[i + 1]])

    for idx, checkpoint_path in enumerate(args.checkpoint):
        print(f"\nProcessing checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        ckpt_args = checkpoint.get("args", {})

        seed = ckpt_args.get("seed", 0)
        set_seeds(seed)

        train_file, val_file, batch_size, num_workers = resolve_data_args(args, checkpoint)

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

        # Print the summary only for the first checkpoint to avoid repeated output spam
        if idx == 0:
            summary_input = first_batch[:, :-1, :].to(device)
            summary(
                model,
                input_data=[summary_input],
                col_names=["input_size", "output_size", "num_params", "params_percent", "mult_adds", "trainable"],
            )

        original_np, generated_np = generate_original_and_generated(
            model=model,
            loader=loader,
            device=device,
            max_batches=args.max_batches,
        )

        log_dir = infer_log_dir_from_checkpoint(checkpoint_path)
        txt_meta = parse_arguments_txt(os.path.join(log_dir, "arguments.txt"))
        caption = build_run_caption(txt_meta)

        if original_shared is None:
            original_shared = original_np
        
        # Create one dedicated subdirectory for this checkpoint and reuse quickLundPlot
        run_plot_dir = os.path.join(out_dir, f"run_{idx+1}")
        os.makedirs(run_plot_dir, exist_ok=True)

        quickLundPlot(
            inputs=[original_np, generated_np],
            labels=["original", caption],
            out_dir=run_plot_dir,
            hist2d_range=[args.hist2d_xrange, args.hist2d_yrange],
            hist1d_ranges=hist1d_ranges,
            hist2d_bins=(30, 40),
            hist1d_bins=args.hist1d_bins,
        )

        generated_inputs.append(generated_np)
        generated_labels.append(caption)

        loss_curves_csv = load_loss_history_csv(os.path.join(log_dir, "loss_history.csv"))

        run_infos.append(
            {
                "checkpoint_path": checkpoint_path,
                "log_dir": log_dir,
                "caption": caption,
                "txt_meta": txt_meta,
                "loss_curves_csv": loss_curves_csv,
            }
        )


    # Reuse the existing plotting utility from train_ublund.py
    # Plot one shared original plus all generated samples together
    combined_inputs = [original_shared] + generated_inputs
    combined_labels = ["original"] + generated_labels

    if not args.not_save_both_hist1d_scales:
        # Save the linear-scale histogram first
        plot_combined_1dhist(
            inputs=combined_inputs,
            labels=combined_labels,
            out_dir=out_dir,
            hist1d_ranges=hist1d_ranges,
            hist1d_bins=args.hist1d_bins,
            logy=False,
            out_name="projection_combined",
        )

        # Save the standard log-scale histogram with the current clamped lower bound
        plot_combined_1dhist(
            inputs=combined_inputs,
            labels=combined_labels,
            out_dir=out_dir,
            hist1d_ranges=hist1d_ranges,
            hist1d_bins=args.hist1d_bins,
            logy=True,
            out_name="projection_combined_logy",
            logy_floor_mode="clamped",
        )

        # Save an additional log-scale histogram that exposes the sparse tail
        plot_combined_1dhist(
            inputs=combined_inputs,
            labels=combined_labels,
            out_dir=out_dir,
            hist1d_ranges=hist1d_ranges,
            hist1d_bins=args.hist1d_bins,
            logy=True,
            out_name="projection_combined_logy_tail",
            logy_floor_mode="tail",
        )
    else:
        plot_combined_1dhist(
            inputs=combined_inputs,
            labels=combined_labels,
            out_dir=out_dir,
            hist1d_ranges=hist1d_ranges,
            hist1d_bins=args.hist1d_bins,
            logy=args.hist1d_logy,
            out_name="projection_combined_logy" if args.hist1d_logy else "projection_combined",
            logy_floor_mode="clamped",
        )

    # Overlay loss curves from all runs using loss_history.csv
    plot_combined_losses(
        run_infos=run_infos,
        out_dir=out_dir,
    )

    print(f"Plots written to: {out_dir}")
    print("Processed runs:")
    for info in run_infos:
        print(f"  - {info['caption']}  [{info['checkpoint_path']}]")


if __name__ == "__main__":
    main()
