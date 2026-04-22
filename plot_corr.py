#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
First-vs-Second Emission Correlation (2D hist) for log(kt) and log(1/ΔR)
with smart branch picking + per-figure auto plot ranges + correlation in title.

Auto-range (when --xrange/--yrange are not given):
  - Use symmetric quantiles [q, 1-q] (default q=0.005) on finite data
  - Add padding (default 3%) around that interval
  - Done separately for each figure so every plot gets its own best range

Requirements:
  pip install uproot awkward numpy matplotlib tqdm
"""

import argparse
from pathlib import Path
import re
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="For each jet, take the 1st and 2nd emissions and plot 2D hist correlations "
                    "for log(kt) and log(1/ΔR), with smart branch picking and auto ranges."
    )
    # Input
    p.add_argument("--file_list", type=str, default="fileList.txt",
                   help="Text file of ROOT paths (one per line).")
    p.add_argument("--tree", type=str, default="auto",
                   help="TTree name; 'auto' will pick the first TTree found.")

    # Branch names (use 'auto' to let the script choose by pattern)
    p.add_argument("--branch_kt", type=str, default="auto",
                   help="Per-jet list branch for kt, or 'auto' to infer (e.g. '*_kt' or 'var2').")
    p.add_argument("--branch_dR", type=str, default="auto",
                   help="Per-jet list branch for deltaR, or 'auto' to infer (e.g. '*deltaR' or 'var1').")

    # If your branches are already logs, set these flags
    p.add_argument("--kt_is_log", action="store_true",
                   help="Selected kt-branch already stores log(kt).")
    p.add_argument("--dR_is_loginv", action="store_true",
                   help="Selected deltaR-branch already stores log(1/ΔR).")

    # Binning / ranges
    p.add_argument("--xbins", type=int, default=60)
    p.add_argument("--ybins", type=int, default=60)
    p.add_argument("--xrange", type=float, nargs=2, default=None,
                   help="X range (xmin xmax); if omitted, auto by data quantiles.")
    p.add_argument("--yrange", type=float, nargs=2, default=None,
                   help="Y range (ymin ymax); if omitted, auto by data quantiles.")

    # Auto-range tuning (only used when no --xrange/--yrange provided)
    p.add_argument("--auto_q", type=float, default=0.005,
                   help="Tail quantile for auto range (e.g., 0.005 -> use [0.5%, 99.5%]).")
    p.add_argument("--auto_pad", type=float, default=0.03,
                   help="Extra padding ratio around the quantile range (e.g., 0.03 -> 3%).")

    # Normalization (for the color scale)
    p.add_argument("--norm", type=str, default="per_jet2",
                   choices=["none", "per_jet2", "per_pair"],
                   help="Color normalization: "
                        "'none' raw counts; "
                        "'per_jet2' divide by #jets with >=2 emissions (default); "
                        "'per_pair' divide by total pair count in-range.")

    # Output
    p.add_argument("--out_dir", type=str, default="plots_corr2d",
                   help="Where to save plots.")
    p.add_argument("--out_tag", type=str, default="corr2d",
                   help="Output prefix; final names: <tag>_<filestem>_logkt.png / _logDRinv.png")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--cmap", type=str, default="viridis")

    # Misc
    p.add_argument("--debug", action="store_true")
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


def _collect_ttrees(node) -> list:
    trees = []
    for k, cls in node.classnames().items():
        base = k.split(";")[0]
        obj = node[k]
        if cls == "TTree":
            trees.append(base)
        elif cls in ("TDirectoryFile", "TDirectory"):
            trees.extend([f"{base}/{t}" for t in _collect_ttrees(obj)])
    return trees


def _pick_tree(f, name: str):
    if name != "auto":
        return f[name], name
    trees = _collect_ttrees(f)
    if not trees:
        raise KeyError("No TTree found.")
    return f[trees[0]], trees[0]


def _is_perjet_list_numeric(tree, bname: str, sample_events: int = 64) -> bool:
    """True if branch is a per-jet list (axis=1 exists) and numeric."""
    try:
        n = getattr(tree, "num_entries", 0) or 0
        entry_stop = min(n, sample_events) if n > 0 else None
        arr = tree[bname].array(library="ak", entry_stop=entry_stop)
        # must have an inner list axis
        _ = ak.num(arr, axis=1)
        flat = ak.flatten(arr, axis=None)
        np_arr = ak.to_numpy(flat)
        return np_arr.size == 0 or (np_arr.dtype is not None and np_arr.dtype.kind in ("f", "i", "u"))
    except Exception:
        return False


def _score_branch_for_role(name: str, role: str) -> int:
    """
    Heuristic score per role ('kt' or 'dR').
    Rules:
      - '*_kt' -> kt
      - '*deltaR' -> dR
      - 'var1' default dR, 'var2' default kt
    """
    s = name.lower()
    score = 0
    if role == "kt":
        if s.endswith("_kt") or s.endswith("kt"):
            score += 100
        if "logkt" in s or "log_kt" in s:
            score += 15
        if s == "var2":
            score += 80
        if "kt" in s:
            score += 10
    elif role == "dR":
        if s.endswith("deltar") or "deltar" in s:
            score += 100
        if re.search(r"(?:^|[_\-])dr(?:$|[_\-])", s):
            score += 20
        if s == "var1":
            score += 80
    return score


def _auto_pick_branches(tree, explicit_kt: str, explicit_dR: str, debug=False):
    """Auto-decide kt & deltaR branches unless explicitly given."""
    names = list(tree.keys())

    bkt = explicit_kt if (explicit_kt != "auto" and explicit_kt in names) else None
    bdR = explicit_dR if (explicit_dR != "auto" and explicit_dR in names) else None

    cand = [b for b in names if _is_perjet_list_numeric(tree, b)]
    if debug:
        print(f"[debug] per-jet list numeric candidates: {cand}")

    if bkt is None:
        scored = sorted(cand, key=lambda n: _score_branch_for_role(n, "kt"), reverse=True)
        bkt = scored[0] if scored and _score_branch_for_role(scored[0], "kt") > 0 else None
    if bdR is None:
        scored = sorted(cand, key=lambda n: _score_branch_for_role(n, "dR"), reverse=True)
        bdR = scored[0] if scored and _score_branch_for_role(scored[0], "dR") > 0 else None

    if bkt is not None and bdR == bkt:
        scored = sorted(cand, key=lambda n: _score_branch_for_role(n, "dR"), reverse=True)
        for n in scored:
            if n != bkt and _score_branch_for_role(n, "dR") > 0:
                bdR = n
                break

    if bkt is None or bdR is None:
        raise RuntimeError("Failed to auto-pick branches. "
                           "Try passing --branch_kt and --branch_dR explicitly.")

    if debug:
        print(f"[debug] picked branches -> kt: '{bkt}', deltaR: '{bdR}'")
    return bkt, bdR


def _ensure_two_emissions_mask(a: ak.Array) -> np.ndarray:
    """Return boolean mask of jets with at least 2 emissions for array a."""
    try:
        counts = ak.to_numpy(ak.num(a, axis=1))
        return counts >= 2
    except Exception:
        raise ValueError("Branch is not a per-jet list; need a jagged/list array to take [0] and [1].")


def _to_log_pairs(a_dr, a_kt, kt_is_log=False, dR_is_loginv=False):
    """
    From per-jet lists, extract first & second emissions and map to:
      (logkt1, logkt2), (logDRinv1, logDRinv2)
    Returns: (xk, yk), (xd, yd) as flat numpy arrays (after finite mask), and jets>=2 count
    """
    m1 = _ensure_two_emissions_mask(a_dr)
    m2 = _ensure_two_emissions_mask(a_kt)
    m = m1 & m2

    dr1 = ak.to_numpy(a_dr[m, 0])
    dr2 = ak.to_numpy(a_dr[m, 1])
    kt1 = ak.to_numpy(a_kt[m, 0])
    kt2 = ak.to_numpy(a_kt[m, 1])

    eps = 1e-12

    if kt_is_log:
        logkt1, logkt2 = kt1, kt2
    else:
        logkt1 = np.log(np.clip(kt1, eps, None))
        logkt2 = np.log(np.clip(kt2, eps, None))

    if dR_is_loginv:
        logDRinv1, logDRinv2 = dr1, dr2
    else:
        logDRinv1 = np.log(1.0 / np.clip(dr1, eps, None))
        logDRinv2 = np.log(1.0 / np.clip(dr2, eps, None))

    mk = np.isfinite(logkt1) & np.isfinite(logkt2)
    md = np.isfinite(logDRinv1) & np.isfinite(logDRinv2)

    return (logkt1[mk], logkt2[mk]), (logDRinv1[md], logDRinv2[md]), int(m.sum())


def _compute_auto_ranges(x, y, xrange_arg, yrange_arg, q=0.005, pad=0.03):
    """
    Decide (xrange, yrange) for a single figure.
    - If user provided xrange_arg/yrange_arg, use them directly.
    - Else, compute by quantiles [q, 1-q] of finite data, then expand by 'pad'.
    """
    def _auto(a):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (-1.0, 1.0)
        lo = np.quantile(a, q)
        hi = np.quantile(a, 1.0 - q)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            # fallback to min/max with small padding
            lo, hi = np.nanmin(a), np.nanmax(a)
        span = hi - lo if hi > lo else max(abs(hi), 1.0)
        lo -= pad * span
        hi += pad * span
        return (float(lo), float(hi))

    xr = tuple(xrange_arg) if xrange_arg is not None else _auto(x)
    yr = tuple(yrange_arg) if yrange_arg is not None else _auto(y)
    return xr, yr


def _plot_2d(x, y, xbins, ybins, xrange_arg, yrange_arg, auto_q, auto_pad,
             norm, n_jets2, title, cmap, out_png, dpi):
    # Decide ranges per figure (auto if not provided)
    xr, yr = _compute_auto_ranges(x, y, xrange_arg, yrange_arg, q=auto_q, pad=auto_pad)

    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[xbins, ybins],
        range=[xr, yr]
    )

    if norm == "per_jet2":
        Hn = H / float(n_jets2 if n_jets2 > 0 else 1.0)
        zlabel = "counts per jet (≥2 emissions)"
    elif norm == "per_pair":
        s = H.sum()
        Hn = H if s == 0 else H / s
        zlabel = "probability per pair"
    else:
        Hn = H
        zlabel = "counts"

    # --- NEW: compute Pearson correlation for (x, y) and append to title ---
    if x.size >= 2 and y.size >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
        # Guard against zero-variance vectors
        if np.std(x) > 0 and np.std(y) > 0:
            corr = float(np.corrcoef(x, y)[0, 1])
            corr_str = f"{corr:.4f}"
        else:
            corr_str = "N/A"
    else:
        corr_str = "N/A"
    final_title = f"{title['title']}\ncorrelation = {corr_str}"

    # Plot
    plt.figure(figsize=(7.6, 6.4))
    mesh = plt.pcolormesh(xedges, yedges, Hn.T, shading="auto", cmap=cmap)
    cbar = plt.colorbar(mesh)
    cbar.set_label(zlabel)
    plt.xlabel(title["xlabel"])
    plt.ylabel(title["ylabel"])
    plt.title(final_title)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    return int(H.sum()), xr, yr


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    files = read_file_list(args.file_list)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fpath in tqdm(files, desc="Files"):
        fpath = Path(fpath)
        with uproot.open(fpath) as uf:
            tree, tpath = _pick_tree(uf, args.tree)

            # ---- pick branches (auto or explicit) ----
            bkt, bdR = _auto_pick_branches(
                tree, args.branch_kt, args.branch_dR, debug=args.debug
            )

            # Read per-jet list branches
            a_kt = tree[bkt].array(library="ak")
            a_dR = tree[bdR].array(library="ak")

            # Build pairs & logs
            (xk, yk), (xd, yd), n_jets2 = _to_log_pairs(
                a_dR, a_kt,
                kt_is_log=args.kt_is_log,
                dR_is_loginv=args.dR_is_loginv
            )

            # Plot: log(kt1) vs log(kt2)
            kt_title = {
                "title": f"log(kt): 1st vs 2nd emission\n{fpath.stem}"
                         f" | jets≥2={n_jets2} | norm={args.norm}\n(kt='{bkt}', dR='{bdR}')",
                "xlabel": "log(kt₁)",
                "ylabel": "log(kt₂)"
            }
            out_png_kt = str(out_dir / f"{args.out_tag}_{fpath.stem}_logkt_corr.png")
            total_pairs_kt, xr_kt, yr_kt = _plot_2d(
                xk, yk, args.xbins, args.ybins,
                args.xrange, args.yrange, args.auto_q, args.auto_pad,
                args.norm, n_jets2, kt_title, args.cmap, out_png_kt, args.dpi
            )

            # Plot: log(1/ΔR1) vs log(1/ΔR2)
            dr_title = {
                "title": f"log(1/ΔR): 1st vs 2nd emission\n{fpath.stem}"
                         f" | jets≥2={n_jets2} | norm={args.norm}\n(kt='{bkt}', dR='{bdR}')",
                "xlabel": "log(1/ΔR₁)",
                "ylabel": "log(1/ΔR₂)"
            }
            out_png_dr = str(out_dir / f"{args.out_tag}_{fpath.stem}_logDRinv_corr.png")
            total_pairs_dr, xr_dr, yr_dr = _plot_2d(
                xd, yd, args.xbins, args.ybins,
                args.xrange, args.yrange, args.auto_q, args.auto_pad,
                args.norm, n_jets2, dr_title, args.cmap, out_png_dr, args.dpi
            )

            print(f"[done] {fpath.name}: saved\n"
                  f"  {out_png_kt} (pairs={total_pairs_kt}, xrange={xr_kt}, yrange={yr_kt})\n"
                  f"  {out_png_dr} (pairs={total_pairs_dr}, xrange={xr_dr}, yrange={yr_dr})\n"
                  f"  picked kt='{bkt}', deltaR='{bdR}'")

if __name__ == "__main__":
    main()
