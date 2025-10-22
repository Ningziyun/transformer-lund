#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot loss function vs. epoch from ROOT files listed in a text file.

Default behavior:
- If a file has only a single loss branch (no epoch), synthesize epochs as 1..N.
- If a file has both epoch and loss branches, use the real epoch array.

You can still override with --epoch_branch / --loss_branch.

Requirements:
    pip install uproot awkward numpy matplotlib
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker


# ---------------------------
# Argument parsing
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Plot loss vs. epoch from ROOT files")
    p.add_argument("--file_list", type=str, default="fileList.txt",
                   help="Text file listing ROOT file paths (one per line).")
    p.add_argument("--tree", type=str, default="auto",
                   help="TTree name; 'auto' picks the first TTree found in each file.")
    p.add_argument("--epoch_branch", type=str, default="auto",
                   help="Branch name for epoch; 'auto' tries to guess; if not found, use 1..N index.")
    p.add_argument("--loss_branch", type=str, default="auto",
                   help="Branch name for loss; 'auto' tries to guess.")
    p.add_argument("--out_dir", type=str, default="plots_loss",
                   help="Directory to save the plot(s).")
    p.add_argument("--out_name", type=str, default="loss_vs_epoch.png",
                   help="Output filename (PNG).")
    p.add_argument("--xrange", type=float, nargs=2, default=None,
                   help="X range (xmin xmax).")
    p.add_argument("--yrange", type=float, nargs=2, default=None,
                   help="Y range (ymin ymax).")
    p.add_argument("--xscale", type=str, choices=["linear", "log"], default="linear",
                   help="Scale for x-axis.")
    p.add_argument("--yscale", type=str, choices=["linear", "log"], default="linear",
                   help="Scale for y-axis.")
    p.add_argument("--smooth", type=int, default=0,
                   help="Optional moving-average window size (N>1) applied to loss.")
    p.add_argument("--dpi", type=int, default=200,
                   help="Figure DPI.")
    p.add_argument("--epoch_start", type=float, default=1.0,
                   help="When synthesizing epochs, start value (default 1).")
    p.add_argument("--epoch_step", type=float, default=1.0,
                   help="When synthesizing epochs, step size (default 1).")

    # ---- Title controls (normal & rich) ----
    p.add_argument("--title", type=str, default="Loss vs. Epoch",
                   help="Figure title (ignored if --title_rich is provided).")
    p.add_argument("--title_color", type=str, default="Black",
                   help="Color for the whole title (ignored if --title_rich is provided).")
    p.add_argument("--title_size", type=float, default=None,
                   help="Font size for the normal title and as default for rich segments.")

    # title_rich: paragraph + line break + automatic scaling + automatic white space
    p.add_argument("--title_rich", type=str, default=None,
                   help=("Rich title: segments separated by ';'. "
                         "Each segment is 'text|color|size|weight'. "
                         "Use '\\n' or '<br>' (as a standalone segment) to break line."))
    p.add_argument("--title_fig_y", type=float, default=0.985,
                   help="Vertical anchor in figure coords (0~1) for the top of the rich title.")
    p.add_argument("--title_seg_sep", type=float, default=0.0,
                   help="Horizontal spacing between segments in the same line.")
    p.add_argument("--title_line_sep", type=float, default=2.0,
                   help="Vertical spacing between lines.")
    p.add_argument("--title_width_max_frac", type=float, default=0.95,
                   help="Max allowed title width as a fraction of figure width before downscaling.")
    p.add_argument("--title_top_pad_frac", type=float, default=0.015,
                   help="Extra top padding fraction reserved above axes for the title height.")
    p.add_argument("--title_min_size", type=float, default=6.0,
                   help="Lower bound for autoscaled font size to avoid disappearing text.")
    return p.parse_args()


# ---------------------------
# Helpers (I/O and arrays)
# ---------------------------
def read_file_list(path_txt: str) -> list:
    p = Path(path_txt)
    if not p.exists():
        raise FileNotFoundError(f"file_list not found: {p}")
    files = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            files.append(s)
    if not files:
        raise RuntimeError(f"No valid file paths in {p}")
    return files


def first_ttree_name(uf) -> str:
    for k, v in uf.classnames().items():
        if v == "TTree":
            return k
    raise KeyError("No TTree found in file.")


def list_array_like_names(tree):
    names = []
    for b in tree.keys():
        try:
            arr = tree[b].array(library="ak", entry_stop=64)
            _ = ak.flatten(arr, axis=None)
            names.append(b)
        except Exception:
            continue
    return names


def autopick_epoch_loss(tree, epoch_hint="auto", loss_hint="auto") -> Tuple[Optional[str], str]:
    keys = list(tree.keys())
    arrays = list_array_like_names(tree)

    if epoch_hint != "auto" and loss_hint != "auto":
        return epoch_hint, loss_hint

    def find_epoch() -> Optional[str]:
        for n in keys:
            ln = n.lower()
            if any(s in ln for s in ("epoch", "step", "iter", "global_step")):
                return n
        return None

    def find_loss() -> Optional[str]:
        for n in keys:
            ln = n.lower()
            if any(s in ln for s in ("val_loss", "train_loss", "loss_fn", "lossfunction", "loss")):
                return n
        return arrays[0] if arrays else None

    epoch_name = epoch_hint if epoch_hint != "auto" else find_epoch()
    loss_name = loss_hint if loss_hint != "auto" else find_loss()

    if loss_name is None:
        raise RuntimeError(f"Could not find a loss-like branch in: {keys}")

    if epoch_name == loss_name:
        epoch_name = None

    return epoch_name, loss_name


def to_aligned_1d(epoch_arr: ak.Array, loss_arr: ak.Array) -> Tuple[np.ndarray, np.ndarray]:
    def is_jagged(a):
        try:
            _ = ak.num(a, axis=1)
            return True
        except Exception:
            return False

    e_jag = is_jagged(epoch_arr)
    l_jag = is_jagged(loss_arr)

    if e_jag and l_jag:
        if ak.any(ak.num(epoch_arr) != ak.num(loss_arr)):
            raise ValueError("Jagged epoch/loss lengths do not match per entry.")
        e = ak.flatten(epoch_arr, axis=None)
        l = ak.flatten(loss_arr, axis=None)
        return ak.to_numpy(e), ak.to_numpy(l)

    if (not e_jag) and (not l_jag):
        e = ak.to_numpy(epoch_arr)
        l = ak.to_numpy(loss_arr)
        if e.shape != l.shape:
            raise ValueError(f"Flat arrays have different shapes: epoch {e.shape} vs loss {l.shape}")
        return e, l

    raise ValueError("Cannot align epoch/loss: one is jagged and the other is flat.")


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return y
    w = int(window)
    if w <= 1 or w >= len(y):
        return y
    kernel = np.ones(w, dtype=float) / w
    smoothed = np.convolve(y, kernel, mode="valid")
    left = (len(y) - len(smoothed)) // 2
    right = len(y) - len(smoothed) - left
    return np.pad(smoothed, (left, right), mode="edge")


def apply_axes(xvals_all, args):
    if args.xrange:
        plt.xlim(args.xrange)
    if args.yrange:
        plt.ylim(args.yrange)
    ax = plt.gca()
    if args.xscale == "log":
        if all((len(x) == 0) or np.all(np.asarray(x) > 0) for x in xvals_all):
            ax.set_xscale("log")
        else:
            print("[warn] xscale=log requested, but some epochs <= 0; keeping linear x-scale.")
    if args.yscale == "log":
        ax.set_yscale("log")


# ---------------------------
# Rich title helpers
# ---------------------------
def _parse_rich_spec(spec: str, default_size: Optional[float]) -> List[List[Dict]]:
    """Parse 'text|color|size|weight' segments into lines of segment dicts."""
    def is_newline(tok: str) -> bool:
        t = tok.strip().lower()
        return t in ("\\n", "<br>", "")

    lines, cur = [], []
    for raw in spec.split(";"):
        tok = raw.strip()
        if is_newline(tok):
            if cur or not lines:
                lines.append(cur)
                cur = []
            continue

        parts = tok.split("|")
        seg = {
            "text":   parts[0] if len(parts) > 0 else "",
            "color":  parts[1] if len(parts) > 1 and parts[1] else None,
            "size":   float(parts[2]) if len(parts) > 2 and parts[2] else default_size,
            "weight": parts[3] if len(parts) > 3 and parts[3] else None,
        }
        cur.append(seg)

    if cur:
        lines.append(cur)
    return lines


def _build_title_box(lines: List[List[Dict]], seg_sep: float, line_sep: float, size_scale: float = 1.0,
                     min_size: float = 6.0):
    """Create (OffsetBox, list of final sizes) from parsed lines with a global size scale."""
    line_boxes = []
    for line in lines:
        seg_areas = []
        for seg in line:
            props = {}
            if seg.get("color"):  props["color"] = seg["color"]
            if seg.get("weight"): props["weight"] = seg["weight"]
            if seg.get("size") is not None:
                props["size"] = max(min_size, float(seg["size"]) * size_scale)
            seg_areas.append(TextArea(seg.get("text", ""), textprops=props))
        line_boxes.append(HPacker(children=seg_areas, align="center", pad=0, sep=seg_sep))
    box = line_boxes[0] if len(line_boxes) == 1 else VPacker(children=line_boxes, align="center", pad=0, sep=line_sep)
    return box


def add_rich_title_figure(fig,
                          spec: str,
                          default_size: Optional[float],
                          y: float,
                          seg_sep: float,
                          line_sep: float,
                          width_max_frac: float,
                          top_pad_frac: float,
                          min_size: float):
    """
    Add a multi-colored, auto-fitting title *in figure coordinates*, centered on top.
    - Autoshrinks font sizes uniformly if the title is wider than width_max_frac of the figure.
    - Reserves enough top margin for the title height to avoid overlap with axes.
    """
    lines = _parse_rich_spec(spec, default_size)
    if not lines:
        return

    # 1) Build once and measure
    box = _build_title_box(lines, seg_sep, line_sep, size_scale=1.0, min_size=min_size)
    anch = AnchoredOffsetbox(loc="upper center", child=box, pad=0.0,
                             bbox_to_anchor=(0.5, y), bbox_transform=fig.transFigure, frameon=False)
    fig.add_artist(anch)

    fig.canvas.draw()  # get a renderer
    bb = box.get_window_extent(fig.canvas.get_renderer())
    fig_w, fig_h = fig.bbox.width, fig.bbox.height
    width_frac = bb.width / fig_w

    # 2) If too wide → downscale uniformly and rebuild
    if width_frac > width_max_frac:
        scale = width_max_frac / max(width_frac, 1e-9)
        fig.artists.remove(anch)  # remove old
        box = _build_title_box(lines, seg_sep, line_sep, size_scale=scale, min_size=min_size)
        anch = AnchoredOffsetbox(loc="upper center", child=box, pad=0.0,
                                 bbox_to_anchor=(0.5, y), bbox_transform=fig.transFigure, frameon=False)
        fig.add_artist(anch)
        fig.canvas.draw()
        bb = box.get_window_extent(fig.canvas.get_renderer())

    # 3) Reserve vertical space for the title (avoid overlap / clipping)
    height_frac = bb.height / fig_h
    top_rect = max(0.0, 1.0 - height_frac - top_pad_frac)
    plt.tight_layout(rect=[0, 0, 1, top_rect])


# ---------------------------
# Main plotting
# ---------------------------
def main():
    args = parse_args()
    files = read_file_list(args.file_list)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    fig = plt.figure(figsize=(8.0, 6.2))
    ax = plt.gca()
    xvals_all = []

    for fpath in files:
        fpath = fpath.strip()
        with uproot.open(fpath) as uf:
            tree_name = first_ttree_name(uf) if args.tree == "auto" else args.tree
            tree = uf[tree_name]

            epoch_name, loss_name = autopick_epoch_loss(
                tree,
                epoch_hint=args.epoch_branch,
                loss_hint=args.loss_branch,
            )

            loss_arr = tree[loss_name].array(library="ak")
            y = ak.to_numpy(ak.flatten(loss_arr, axis=None))

            if (epoch_name is None) or (epoch_name == loss_name):
                x = args.epoch_start + np.arange(len(y), dtype=float) * args.epoch_step
                label = f"{Path(fpath).stem} [index->{loss_name}]"
            else:
                epoch_arr = tree[epoch_name].array(library="ak")
                x, y = to_aligned_1d(epoch_arr, loss_arr)
                label = f"{Path(fpath).stem} [{epoch_name}->{loss_name}]"

            order = np.argsort(x)
            x = x[order]; y = y[order]

            if args.smooth and args.smooth > 1:
                y = moving_average(y, args.smooth)

            if args.yscale == "log":
                mask = y > 0
                if not np.any(mask):
                    print(f"[warn] All non-positive loss in {fpath}; skipping in log scale.")
                    continue
                x, y = x[mask], y[mask]

            ax.plot(x, y, label=label)
            xvals_all.append(x)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    apply_axes(xvals_all, args)

    # ---- Title rendering ----
    if args.title_rich:
        # place in figure coords + autoscale + reserve space
        add_rich_title_figure(
            fig,
            spec=args.title_rich,
            default_size=args.title_size,
            y=args.title_fig_y,
            seg_sep=args.title_seg_sep,
            line_sep=args.title_line_sep,
            width_max_frac=args.title_width_max_frac,
            top_pad_frac=args.title_top_pad_frac,
            min_size=args.title_min_size,
        )
    else:
        ax.set_title(args.title, color=args.title_color, fontsize=args.title_size)
        plt.tight_layout()

    ax.legend()
    plt.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"[OK] Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
