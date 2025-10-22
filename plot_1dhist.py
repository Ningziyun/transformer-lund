#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare kt/deltaR or pt/eta/phi distributions across multiple ROOT files.

Modes:
- kdr (default): plot Log(kt) and Log(1/deltaR)
- kin:           plot pt, eta, phi

New:
- --mode {kdr,kin}: choose which set of plots to produce.
- Kinematics (kin) auto picking:
    * pt : name contains 'pt' or name == 'var1' or the 1st array-like branch
    * eta: name contains 'eta' or name == 'var2' or the 2nd array-like branch
    * phi: name contains 'phi' or name == 'var3' or the 3rd array-like branch
- --max_emissions: limit the maximum number of emissions PER JET by truncating
  each jagged array to its first N entries (applied to all branches used).

Existing features:
- Auto branch picking for kdr mode: 'kt' or 'var2' -> kt; 'deltaR' or 'var1' -> deltaR
- Per-jet normalization (counts / number of jets)
- 1D histogram plotted as line (step plot) with legend
- Legends and titles can be customized
- --xrange / --yrange for plot ranges
- --xscale / --yscale for linear/log axes with safeguards

Requirements:
  pip install uproot awkward numpy matplotlib
"""

import argparse
from pathlib import Path
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt


# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compare distributions across ROOT files")
    p.add_argument("--file_list", type=str, default="fileList.txt",
                   help="Text file of ROOT paths (one per line).")
    p.add_argument("--tree", type=str, default="auto",
                   help="TTree name; 'auto' picks the first TTree found.")
    p.add_argument("--bins", type=int, default=50, help="Number of bins for histograms.")
    p.add_argument("--out_dir", type=str, default="plots_1d", help="Where to save plots.")

    # Mode: kdr (Log(kt)/Log(1/deltaR)) or kin (pt/eta/phi)
    p.add_argument("--mode", type=str, choices=["kdr", "kin"], default="kdr",
                   help="Plot set: 'kdr' for Log(kt)/Log(1/deltaR) (default), 'kin' for pt/eta/phi.")

    # Titles
    p.add_argument("--title_kt", type=str, default="kt distributions", help="Title for kt plot (kdr mode).")
    p.add_argument("--title_dR", type=str, default="deltaR distributions", help="Title for deltaR plot (kdr mode).")
    p.add_argument("--title_pt", type=str, default="pT distributions", help="Title for pt plot (kin mode).")
    p.add_argument("--title_eta", type=str, default="eta distributions", help="Title for eta plot (kin mode).")
    p.add_argument("--title_phi", type=str, default="phi distributions", help="Title for phi plot (kin mode).")

    # Axis ranges (apply to all figures within the chosen mode)
    p.add_argument("--xrange", type=float, nargs=2, default=None,
                   help="X range (xmin xmax).")
    p.add_argument("--yrange", type=float, nargs=2, default=None,
                   help="Y range (ymin ymax).")

    # Axis scales
    p.add_argument("--xscale", type=str, choices=["linear", "log"], default="linear",
                   help="Scale for x-axis (applies to all figures).")
    p.add_argument("--yscale", type=str, choices=["linear", "log"], default="linear",
                   help="Scale for y-axis (applies to all figures).")

    # Emission cap (per-jet)
    p.add_argument("--max_emissions", "--maxN",type=int, default=None,
                   help="If set, truncate each jet's emissions to the first N entries (per-jet).")

    return p.parse_args()


# ---------------------------
# Helpers
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


def _first_ttree_name(uf) -> str:
    """Return the first TTree key in a ROOT file."""
    for k, v in uf.classnames().items():
        if v == "TTree":
            return k
    raise KeyError("No TTree found in file.")


def pick_branches_kdr(tree_keys):
    """
    Decide kt and deltaR branches based on simple name heuristics (kdr mode).

      - name contains 'kt'  OR name == 'var2'    -> kt
      - name contains 'deltar' OR name == 'var1' -> deltaR
    """
    bkt, bdR = None, None
    for name in tree_keys:
        lname = name.lower()
        if bkt is None and ("kt" in lname or lname == "var2"):
            bkt = name
        if bdR is None and ("deltar" in lname or lname == "var1"):
            bdR = name
    if bkt is None or bdR is None:
        raise RuntimeError(f"[kdr] Could not auto-pick branches from: {list(tree_keys)}")
    return bkt, bdR


def _array_like(tree, bname):
    """Return awkward array for branch; raises if not readable as array."""
    return tree[bname].array(library="ak")


def _list_array_like_names(tree):
    """
    Return a list of branch names that can be read as (flat or jagged) numeric arrays.
    We try reading a small sample; if it works, we keep it.
    """
    names = []
    for b in tree.keys():
        try:
            arr = tree[b].array(library="ak", entry_stop=32)
            _ = ak.flatten(arr, axis=None)
            names.append(b)
        except Exception:
            continue
    return names


def pick_branches_kin(tree):
    """
    Pick (pt, eta, phi) with priorities:
      pt  : contains 'pt'  or name == 'var1' or 1st array-like
      eta : contains 'eta' or name == 'var2' or 2nd array-like
      phi : contains 'phi' or name == 'var3' or 3rd array-like
    Ensure distinct branches if possible; if collisions, pick next unused.
    """
    keys = list(tree.keys())
    array_like = _list_array_like_names(tree)
    used = set()

    def pick_one(primary_substr, fallback_exact, fallback_index):
        # 1) substring match
        for n in keys:
            ln = n.lower()
            if primary_substr in ln and n not in used:
                return n
        # 2) exact fallback like var1/var2/var3
        for n in keys:
            if n.lower() == fallback_exact and n not in used:
                return n
        # 3) index fallback within array-like list
        if fallback_index < len(array_like):
            for i in range(fallback_index, len(array_like)):
                if array_like[i] not in used:
                    return array_like[i]
        # 4) any remaining
        for n in array_like:
            if n not in used:
                return n
        # 5) give up
        return None

    bpt  = pick_one("pt",  "var1", 0);   used.add(bpt)  if bpt  else None
    beta = pick_one("eta", "var2", 1);   used.add(beta) if beta else None
    bphi = pick_one("phi", "var3", 2);   used.add(bphi) if bphi else None

    if not (bpt and beta and bphi):
        raise RuntimeError(f"[kin] Could not auto-pick pt/eta/phi from branches: {keys}")
    return bpt, beta, bphi


def truncate_emissions(arr, max_emissions):
    """Per-jet truncate to first N emissions if jagged; no-op for flat arrays."""
    if max_emissions is None or max_emissions <= 0:
        return arr
    try:
        _ = ak.num(arr, axis=1)  # probe inner list axis
    except Exception:
        return arr
    return arr[:, :max_emissions]


def compute_hist_perjet(arr, bins):
    """Histogram of flattened emissions divided by number of jets."""
    n_jets = len(arr)
    flat = ak.to_numpy(ak.flatten(arr, axis=None))
    counts, edges = np.histogram(flat, bins=bins)
    return edges, counts / max(n_jets, 1)


def apply_axes_config(centers_all, args):
    """Apply axis ranges and scales safely."""
    if args.xrange:
        plt.xlim(args.xrange)
    if args.yrange:
        plt.ylim(args.yrange)

    ax = plt.gca()
    if args.xscale == "log":
        positive_ok = True
        for c in centers_all:
            if c.size and not np.all(c > 0):
                positive_ok = False
                break
        if positive_ok:
            ax.set_xscale("log")
        else:
            print("[warn] xscale=log requested, but some bin centers are <= 0; keeping xscale=linear.")
    if args.yscale == "log":
        ax.set_yscale("log")


def _plot_one(files, label_fmt, branch_picker, x_label, title, out_path, args):
    """
    Generic overlay plot:
      - branch_picker(tree) -> target branch name
      - label_fmt(stem, bname) -> label string
    """
    plt.figure(figsize=(7.6, 6.4))
    centers_all = []
    for fpath in files:
        with uproot.open(fpath) as uf:
            tree_name = _first_ttree_name(uf) if args.tree == "auto" else args.tree
            tree = uf[tree_name]

            bname = branch_picker(tree)
            arr = _array_like(tree, bname)
            arr = truncate_emissions(arr, args.max_emissions)

            edges, hist = compute_hist_perjet(arr, args.bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            if args.yscale == "log":
                hist = hist.astype(float)
                hist[hist <= 0] = np.nan

            plt.step(centers, hist, where="mid",
                     label=label_fmt(Path(fpath).stem, bname))
            centers_all.append(centers)

    plt.xlabel(x_label)
    plt.ylabel("Emissions per jet")
    plt.title(title)
    apply_axes_config(centers_all, args)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    files = read_file_list(args.file_list)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "kdr":
        # --- kt ---
        def pick_kt(tree):
            bkt, _ = pick_branches_kdr(tree.keys())
            return bkt

        _plot_one(
            files,
            label_fmt=lambda stem, b: f"{stem} [{b}]",
            branch_picker=pick_kt,
            x_label="Log(kt)",
            title=args.title_kt,
            out_path=out_dir / "compare_kt.png",
            args=args,
        )

        # --- deltaR ---
        def pick_dr(tree):
            _, bdR = pick_branches_kdr(tree.keys())
            return bdR

        _plot_one(
            files,
            label_fmt=lambda stem, b: f"{stem} [{b}]",
            branch_picker=pick_dr,
            x_label="Log(1/deltaR)",
            title=args.title_dR,
            out_path=out_dir / "compare_deltaR.png",
            args=args,
        )

    else:  # args.mode == "kin"
        def pick_pt(tree):
            bpt, _, _ = pick_branches_kin(tree)
            return bpt

        def pick_eta(tree):
            _, beta, _ = pick_branches_kin(tree)
            return beta

        def pick_phi(tree):
            _, _, bphi = pick_branches_kin(tree)
            return bphi

        _plot_one(
            files,
            label_fmt=lambda stem, b: f"{stem} [{b}]",
            branch_picker=pick_pt,
            x_label="pT",
            title=args.title_pt,
            out_path=out_dir / "compare_pt.png",
            args=args,
        )

        _plot_one(
            files,
            label_fmt=lambda stem, b: f"{stem} [{b}]",
            branch_picker=pick_eta,
            x_label="eta",
            title=args.title_eta,
            out_path=out_dir / "compare_eta.png",
            args=args,
        )

        _plot_one(
            files,
            label_fmt=lambda stem, b: f"{stem} [{b}]",
            branch_picker=pick_phi,
            x_label="phi",
            title=args.title_phi,
            out_path=out_dir / "compare_phi.png",
            args=args,
        )


if __name__ == "__main__":
    main()
