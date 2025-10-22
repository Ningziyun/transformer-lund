#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay distributions of emissions across multiple ROOT files, with:
  - Auto-detection of two constituent-like branches:
      * kt-like       <- branch name contains ["kt","var2"]
      * deltaR-like   <- branch name contains ["deltar","var1"]
    Fallback: first vector-like branch -> deltaR-like, second -> kt-like.
  - Plot up to a user-selected emission index (k), default = all available.
  - Robust skipping when some files have fewer emissions (no errors).
  - NEW: By default, treat branch values as ALREADY log10(kt) and log10(1/ΔR).
         Only compute logs if --compute_logs is provided.

Outputs (per emission k):
  emission{k}_logkt.png
  emission{k}_log1overdR.png

Requirements:
  pip install uproot awkward numpy matplotlib tqdm
"""

import argparse
from pathlib import Path
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Auto-map branches to kt-like and deltaR-like, then overlay emission-k distributions."
    )
    # IO
    p.add_argument("--file_list", type=str, default="fileList.txt",
                   help="Text file with one ROOT path per line.")
    p.add_argument("--outdir", type=str, default="plots_emissions_auto",
                   help="Directory to save output figures.")
    p.add_argument("--tree_name", type=str, default="auto",
                   help="TTree name. Use 'auto' to detect per file.")
    # Optional manual overrides
    p.add_argument("--branch_kt", type=str, default=None,
                   help="Explicit branch for kt-like (vector-like).")
    p.add_argument("--branch_dr", type=str, default=None,
                   help="Explicit branch for deltaR-like (vector-like).")
    # Emission control
    p.add_argument("--max_emission", type=int, default=0,
                   help="Plot up to this emission index (k). 0 = all available (default).")
    # Binning and ranges
    p.add_argument("--bins", type=int, default=60, help="Number of bins.")
    p.add_argument("--xrange_logkt", type=float, nargs=2, default=None,
                   help="Manual x-range (xmin xmax) for log10(kt). If omitted, auto by quantiles per k.")
    p.add_argument("--xrange_log1dr", type=float, nargs=2, default=None,
                   help="Manual x-range (xmin xmax) for log10(1/deltaR). If omitted, auto by quantiles per k.")
    p.add_argument("--auto_q", type=float, default=0.005,
                   help="Quantile for auto range when --xrange_* not set (symmetric [q,1-q]).")
    p.add_argument("--auto_pad", type=float, default=0.03,
                   help="Padding ratio added around auto quantile range.")
    # Normalization
    p.add_argument("--norm", type=str, default="perjet",
                   choices=["perjet", "pdf", "count"],
                   help="Histogram normalization: 'perjet' divide by #jets(>=k); 'pdf' area=1; 'count' raw counts.")
    # Y-scale
    p.add_argument("--yscale", type=str, default="linear",
                   choices=["linear", "log"],
                   help="Y-axis scale.")
    # Log handling
    p.add_argument("--compute_logs", action="store_true",
                   help="If set, treat branches as raw kt and deltaR and compute log10(kt) and log10(1/dR). "
                        "By default, branches are assumed to already be log-scaled.")
    # Labels and aesthetics
    p.add_argument("--labels", type=str, default=None,
                   help="Comma-separated legend labels for each file (same order as file_list).")
    p.add_argument("--title_suffix", type=str, default="",
                   help="Extra text appended to titles.")
    p.add_argument("--lw", type=float, default=1.8, help="Line width.")
    p.add_argument("--alpha", type=float, default=0.95, help="Line alpha.")
    return p.parse_args()


# ---------------------------
# Helpers
# ---------------------------
def read_file_list(path_txt: str):
    lines = [ln.strip() for ln in Path(path_txt).read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]

def detect_tree_name(fpath: str) -> str:
    """Detect the first TTree found in the file (recursive over subdirs)."""
    with uproot.open(fpath) as f:
        def _iter_tree_keys(directory):
            for k in directory.keys():
                try:
                    obj = directory[k]
                    if hasattr(obj, "classname") and "TTree" in obj.classname:
                        yield k
                except Exception:
                    continue
            for k in directory.keys():
                try:
                    sub = directory[k]
                    if hasattr(sub, "keys"):
                        yield from _iter_tree_keys(sub)
                except Exception:
                    continue

        for key in _iter_tree_keys(f):
            try:
                obj = f[key]
                return obj.object_path
            except Exception:
                continue
    raise RuntimeError(f"No TTree detected in {fpath}")

def open_tree(fpath: str, tree_name: str):
    tn = detect_tree_name(fpath) if tree_name == "auto" else tree_name
    return uproot.open(fpath)[tn]

def list_branches(tree):
    """Return branch names without ';cycle' suffix."""
    return [k.split(";")[0] for k in tree.keys()]

def is_vector_like(tree, br, max_entries=2000):
    """
    Check if branch is vector-like (jagged per jet).
    Consider vector-like if >=10% of entries have len>=1 or any have len>=2.
    """
    try:
        arr = tree.arrays([br], entry_stop=max_entries, library="ak")[br]
        if isinstance(arr, ak.Array):
            lens = ak.num(arr, axis=-1)
            if ak.any(lens >= 2) or (ak.sum(lens >= 1) / max(len(lens), 1) > 0.1):
                return True
    except Exception:
        return False
    return False

def auto_pick_two_constituent_branches(tree, manual_kt=None, manual_dr=None):
    """
    Decide two constituent branches:
      - kt-like     : name contains ["kt","var2"]
      - deltaR-like : name contains ["deltar","var1"]
    Restrict to vector-like branches; fallback to the first two vector-like branches.

    Return: (br_dr, br_kt)   # deltaR-like first, kt-like second
    """
    names = list_branches(tree)
    vec_names = [n for n in names if is_vector_like(tree, n)]
    if not vec_names:
        raise RuntimeError("No vector-like constituent branches found.")

    br_kt = manual_kt if (manual_kt in vec_names) else None
    br_dr = manual_dr if (manual_dr in vec_names) else None

    if br_kt is None:
        for n in vec_names:
            low = n.lower()
            if ("kt" in low) or ("var2" in low):
                br_kt = n
                break
    if br_dr is None:
        for n in vec_names:
            low = n.lower()
            if ("deltar" in low) or ("var1" in low):
                br_dr = n
                break

    picked = []
    if br_dr:
        picked.append(br_dr)
    if br_kt and br_kt != br_dr:
        picked.append(br_kt)

    for n in vec_names:
        if len(picked) >= 2:
            break
        if n not in picked:
            picked.append(n)

    br_dr_final = picked[0]
    br_kt_final = picked[1] if len(picked) > 1 else picked[0]
    return br_dr_final, br_kt_final

def jets_with_ge_k(arr, k: int) -> int:
    """Number of jets having at least k emissions."""
    if k <= 0:
        return len(arr)
    return int(ak.sum(ak.num(arr) >= k))

def kth_emission_numpy(arr, k: int) -> np.ndarray:
    """
    Extract k-th emission (1-based) from a jagged array.
    Returns a 1D numpy array; if some jets have <k, they're skipped.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    mask = ak.num(arr) >= k
    if not ak.any(mask):
        return np.array([], dtype=float)
    return ak.to_numpy(arr[mask][:, k - 1])

def auto_range(datas, q=0.005, pad=0.03):
    """Compute [xmin, xmax] over a list of arrays via symmetric quantiles."""
    if not any(d.size for d in datas):
        return -1.0, 1.0
    concat = np.concatenate([d for d in datas if d.size > 0], dtype=float)
    lo = np.quantile(concat, q)
    hi = np.quantile(concat, 1 - q)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(concat)), float(np.max(concat))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
    span = hi - lo
    return lo - pad * span, hi + pad * span

def make_bins(xmin, xmax, nbins):
    return np.linspace(xmin, xmax, nbins + 1)

def norm_factor(kind: str, counts: np.ndarray, n_jets: int, bin_widths: np.ndarray):
    if kind == "count":
        return 1.0
    if kind == "perjet":
        return 1.0 / max(n_jets, 1)
    if kind == "pdf":
        area = np.sum(counts * bin_widths)
        return 1.0 / area if area > 0 else 1.0
    return 1.0


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    root_files = read_file_list(args.file_list)
    if not root_files:
        raise SystemExit("Empty --file_list.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Legend labels
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(root_files):
            raise SystemExit("--labels count must match number of files.")
    else:
        labels = [Path(f).stem for f in root_files]

    # Store per-file arrays and chosen branches
    dr_arrays = []   # Awkward arrays for ΔR-like branch
    kt_arrays = []   # Awkward arrays for kt-like branch
    picked_pairs = []

    # Read all files & pick branches
    for fpath in tqdm(root_files, desc="Opening trees & picking branches"):
        try:
            tree = open_tree(fpath, args.tree_name)
        except Exception as e:
            raise SystemExit(f"[Error] Opening tree in {fpath}: {e}")

        try:
            br_dr, br_kt = auto_pick_two_constituent_branches(
                tree, manual_kt=args.branch_kt, manual_dr=args.branch_dr
            )
            picked_pairs.append((br_dr, br_kt))
        except Exception as e:
            raise SystemExit(f"[Error] Auto-picking branches in {fpath}: {e}")

        try:
            arr_dr = tree.arrays([br_dr], library="ak")[br_dr]
            arr_kt = tree.arrays([br_kt], library="ak")[br_kt]
        except Exception as e:
            raise SystemExit(f"[Error] Reading branches '{br_dr}'/'{br_kt}' in {fpath}: {e}")

        dr_arrays.append(arr_dr)
        kt_arrays.append(arr_kt)

    # Determine maximum emission index to plot
    def max_emission_in(arr):
        try:
            return int(ak.to_numpy(ak.max(ak.num(arr))).item())
        except Exception:
            return 0

    max_k_dr = max((max_emission_in(a) for a in dr_arrays), default=0)
    max_k_kt = max((max_emission_in(a) for a in kt_arrays), default=0)
    max_k_all = max(max_k_dr, max_k_kt)

    K = min(args.max_emission, max_k_all) if args.max_emission > 0 else max_k_all
    if K <= 0:
        raise SystemExit("No emissions found in provided files.")

    # Loop over emissions k=1..K and make two plots per k
    for k in range(1, K + 1):
        data_logkt = []
        data_log1dr = []
        njet_k_kt = []
        njet_k_dr = []

        for arr_dr, arr_kt in zip(dr_arrays, kt_arrays):
            # denominators: jets with >=k emissions
            njet_k_dr.append(jets_with_ge_k(arr_dr, k))
            njet_k_kt.append(jets_with_ge_k(arr_kt, k))

            # kth values (skip jets with <k)
            dr_k = kth_emission_numpy(arr_dr, k)
            kt_k = kth_emission_numpy(arr_kt, k)

            if args.compute_logs:
                # treat arrays as raw ΔR and kt
                dr_k = dr_k[np.isfinite(dr_k) & (dr_k > 0)]
                kt_k = kt_k[np.isfinite(kt_k) & (kt_k > 0)]
                data_log1dr.append(np.log10(1.0 / dr_k))
                data_logkt.append(np.log10(kt_k))
            else:
                # treat arrays as already log10(1/ΔR) and log10(kt)
                data_log1dr.append(dr_k[np.isfinite(dr_k)])
                data_logkt.append(kt_k[np.isfinite(kt_k)])

        if not any(d.size for d in data_logkt) and not any(d.size for d in data_log1dr):
            print(f"[Info] Emission {k}: no data in any file. Skipping both plots.")
            continue

        # bins/ranges
        if args.xrange_logkt is not None:
            xmin_kt, xmax_kt = args.xrange_logkt
        else:
            xmin_kt, xmax_kt = auto_range(data_logkt, q=args.auto_q, pad=args.auto_pad)

        if args.xrange_log1dr is not None:
            xmin_1dr, xmax_1dr = args.xrange_log1dr
        else:
            xmin_1dr, xmax_1dr = auto_range(data_log1dr, q=args.auto_q, pad=args.auto_pad)

        edges_kt = make_bins(xmin_kt, xmax_kt, args.bins)
        widths_kt = np.diff(edges_kt)
        edges_1dr = make_bins(xmin_1dr, xmax_1dr, args.bins)
        widths_1dr = np.diff(edges_1dr)

        # Plot helper
        def plot_one(datas, njet_list, edges, widths, title, xlabel, fname):
            fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=140)
            has_any = False
            for data, lab, njet in zip(datas, labels, njet_list):
                if data.size == 0 or njet == 0:
                    continue
                counts, _ = np.histogram(data, bins=edges)
                factor = norm_factor(args.norm, counts, njet, widths)
                y = counts * factor
                ax.step(edges[:-1], y, where="post", linewidth=args.lw, alpha=args.alpha, label=lab)
                has_any = True

            if not has_any:
                plt.close(fig)
                print(f"[Info] {fname}: no plottable data. Skipping.")
                return

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel({"count": "Counts", "perjet": "Counts / #Jets(≥k)", "pdf": "PDF (area=1)"}[args.norm])
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_yscale(args.yscale)

            fig.tight_layout()
            out = Path(args.outdir) / f"{fname}.png"
            fig.savefig(out)
            plt.close(fig)
            print(f"[OK] Saved: {out}")

        suf = f" | {args.title_suffix}" if args.title_suffix else ""
        title_kt  = f"log10(kt) - emission {k}{suf}"
        title_1dr = f"log10(1/ΔR) - emission {k}{suf}"

        # Make the two plots for emission k
        plot_one(data_logkt,  njet_k_kt, edges_kt,  widths_kt,  title_kt,  "log10(kt)",    f"emission{k}_logkt")
        plot_one(data_log1dr, njet_k_dr, edges_1dr, widths_1dr, title_1dr, "log10(1/ΔR)", f"emission{k}_log1overdR")

    # Summary of branch picks
    print("\n[Info] Picked branches per file (deltaR-like, kt-like):")
    for f, (br_dr, br_kt) in zip(root_files, picked_pairs):
        print(f"  {f}  ->  ΔR: {br_dr} ; kt: {br_kt}")


if __name__ == "__main__":
    main()
