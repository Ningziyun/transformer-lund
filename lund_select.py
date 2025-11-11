#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read two jagged branches from a ROOT TTree, apply composable operations:

  - shuffle    : per-jet permutation with pair alignment
  - top10      : keep jets with >=10 emissions, then slice [:10]
  - both       : expands to shuffle + top10
  - swap       : swap the two branches' DATA on output (names follow default unless overridden)
  - swapnames  : keep DATA as-is, only swap the OUTPUT NAMES (aliases)

Pipeline order (deterministic when composing):
  shuffle -> top10 -> swap(data) -> swapnames(names)

THEN write them back using **PyROOT** as EXACTLY TWO branches.
By default, the output branch order follows the input order,
unless explicitly overridden with --out_b1 / --out_b2.

Usage examples
--------------
# single ops
python lund_select.py --in rootFiles/log_kt_deltaR.root --out shuffled.root  --mode shuffle --seed 42
python lund_select.py --in rootFiles/log_kt_deltaR.root --out top10.root     --mode top10
python lund_select.py --in rootFiles/log_kt_deltaR.root --out both.root      --mode both --seed 123
python lund_select.py --in rootFiles/log_kt_deltaR.root --out swapped.root   --mode swap
python lund_select.py --in rootFiles/log_kt_deltaR.root --out swappedn.root  --mode swapnames

# compose multiple modes (order is fixed as above)
python lund_select.py --in rootFiles/log_kt_deltaR.root --out shuffled.root  --mode shuffle --seed 42 --mode swap
python lund_select.py --in rootFiles/log_kt_deltaR.root --out top10.root     --mode top10 --mode swap
python lund_select.py --in root.root --out out.root --mode shuffle --mode top10 --mode swapnames
python lund_select.py --in root.root --out out.root --mode both --mode swap

We explicitly create std::vector<float> branches via PyROOT,
so NO auxiliary count/offset branches will appear.

Add a 'cut' mode to filter emissions by x–y rectangular region.

This version extends the original pipeline:
  shuffle -> top10 -> cut -> swap(data) -> swapnames(names)

'cut' keeps only emissions satisfying xmin ≤ x ≤ xmax and ymin ≤ y ≤ ymax,
where x = branch1 (e.g., log_kt) and y = branch2 (e.g., log_1_over_deltaR),
unless swapped by other modes.
"""

import argparse
from typing import Optional, List
import numpy as np
import awkward as ak
import uproot
import ROOT


# ---------------------------
# I/O helpers
# ---------------------------
def autodetect_tree_name(root_path: str) -> str:
    """Find the first TTree inside a ROOT file.
    Split the key at ';' to extract the pure tree name."""
    with uproot.open(root_path) as f:
        for key in f.keys():
            try:
                obj = f[key]
                if getattr(obj, "classname", "") == "TTree":
                    return key.split(";")[0]
            except Exception:
                continue
    raise ValueError("No TTree found in the input ROOT file.")


def read_branches(root_path: str, tree_name: str, in_b1: str, in_b2: str):
    """Read two jagged branches as Awkward arrays.
    Verify that both branches have the same per-jet length."""
    with uproot.open(root_path) as f:
        if tree_name.lower() == "auto":
            tree_name = autodetect_tree_name(root_path)
        tree = f[tree_name]
        arrs = tree.arrays([in_b1, in_b2], library="ak")
    a1 = arrs[in_b1]
    a2 = arrs[in_b2]
    if ak.any(ak.num(a1) != ak.num(a2)):
        raise ValueError("Per-entry lengths of the two branches do not match.")
    return a1, a2, tree_name


# ---------------------------
# Core ops
# ---------------------------
def op_shuffle(a1: ak.Array, a2: ak.Array, seed: Optional[int] = None):
    """Randomly permute emissions within each jet while keeping (a1,a2) aligned.
    Use Awkward-array sorting with random keys for deterministic shuffling."""
    rng = np.random.default_rng(seed)
    counts = ak.num(a1)
    keys_flat = rng.random(int(ak.sum(counts)))
    keys = ak.unflatten(keys_flat, counts)
    perm = ak.argsort(keys, axis=1)
    return a1[perm], a2[perm]


def op_top10(a1: ak.Array, a2: ak.Array):
    """Keep only jets with >=10 emissions, then slice each to first 10.
    Apply a boolean mask followed by jagged slicing."""
    n = ak.num(a1)
    mask = n >= 10
    return a1[mask][:, :10], a2[mask][:, :10]


def op_cut(a1: ak.Array, a2: ak.Array, xmin: float, xmax: float, ymin: float, ymax: float):
    """Filter emissions by rectangular (x,y) window.
    Keep only those emissions satisfying xmin≤x≤xmax and ymin≤y≤ymax."""
    mask_x = (a1 >= xmin) & (a1 <= xmax)
    mask_y = (a2 >= ymin) & (a2 <= ymax)
    mask = mask_x & mask_y
    return a1[mask], a2[mask]


# ---------------------------
# PyROOT writer
# ---------------------------
def write_with_pyroot(out_path: str, tree_name: str, a1: ak.Array, a2: ak.Array, out_b1: str, out_b2: str):
    """Write two jagged arrays as std::vector<float> branches.
    Convert to float32 for compact storage and iterate per jet in PyROOT."""
    a1_32 = ak.values_astype(a1, np.float32)
    a2_32 = ak.values_astype(a2, np.float32)

    fout = ROOT.TFile(out_path, "RECREATE")
    t = ROOT.TTree(tree_name, tree_name)

    v1 = ROOT.std.vector('float')()
    v2 = ROOT.std.vector('float')()
    t.Branch(out_b1, v1)
    t.Branch(out_b2, v2)

    list1 = ak.to_list(a1_32)
    list2 = ak.to_list(a2_32)
    if len(list1) != len(list2):
        raise RuntimeError("Number of jets mismatch after processing.")

    for x1, x2 in zip(list1, list2):
        v1.clear()
        v2.clear()
        for val in x1:
            v1.push_back(float(val))
        for val in x2:
            v2.push_back(float(val))
        t.Fill()

    fout.Write()
    fout.Close()


# ---------------------------
# CLI + main
# ---------------------------
def parse_args():
    """Parse command-line arguments.
    Add new cut options: --cut and four bounds --xmin/xmax/ymin/ymax."""
    p = argparse.ArgumentParser(description="Compose shuffle/top10/cut/swap/swapnames and write two branches via PyROOT.")
    p.add_argument("--in", dest="in_path", required=True, help="Input ROOT file path.")
    p.add_argument("--out", dest="out_path", required=True, help="Output ROOT file path.")
    p.add_argument("--mode", action="append",
                   choices=["shuffle", "top10", "cut", "swap", "swapnames", "both"],
                   required=True,
                   help="Repeatable. Pipeline order: shuffle -> top10 -> cut -> swap -> swapnames.")
    p.add_argument("--tree_name", default="auto", help="TTree name, 'auto' = autodetect first tree.")
    p.add_argument("--in_b1", default="log_kt", help="Input branch 1 name.")
    p.add_argument("--in_b2", default="log_1_over_deltaR", help="Input branch 2 name.")
    p.add_argument("--out_b1", default=None, help="Output branch 1 name.")
    p.add_argument("--out_b2", default=None, help="Output branch 2 name.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for shuffle mode.")

    # --- new cut parameters ---
    p.add_argument("--xmin", type=float, default=None, help="Cut lower bound for x (branch1).")
    p.add_argument("--xmax", type=float, default=None, help="Cut upper bound for x (branch1).")
    p.add_argument("--ymin", type=float, default=None, help="Cut lower bound for y (branch2).")
    p.add_argument("--ymax", type=float, default=None, help="Cut upper bound for y (branch2).")
    return p.parse_args()


def _expand_modes(modes_list: List[str]) -> List[str]:
    """Expand 'both' into ['shuffle','top10'] and preserve fixed pipeline order."""
    selected = set(modes_list)
    if "both" in selected:
        selected.update({"shuffle", "top10"})
    pipeline = ["shuffle", "top10", "cut", "swap", "swapnames"]
    return [m for m in pipeline if m in selected]


def main():
    args = parse_args()
    a1, a2, tname = read_branches(args.in_path, args.tree_name, args.in_b1, args.in_b2)
    modes = _expand_modes(args.mode)

    a1_out, a2_out = a1, a2
    applied = []

    # --- sequentially apply ops in fixed order ---
    if "shuffle" in modes:
        a1_out, a2_out = op_shuffle(a1_out, a2_out, seed=args.seed)
        applied.append("shuffle")

    if "top10" in modes:
        a1_out, a2_out = op_top10(a1_out, a2_out)
        applied.append("top10")

    if "cut" in modes:
        # Ensure all bounds provided
        if None in (args.xmin, args.xmax, args.ymin, args.ymax):
            raise ValueError("cut mode requires --xmin, --xmax, --ymin, --ymax.")
        a1_out, a2_out = op_cut(a1_out, a2_out, args.xmin, args.xmax, args.ymin, args.ymax)
        applied.append(f"cut[{args.xmin},{args.xmax};{args.ymin},{args.ymax}]")

    if "swap" in modes:
        a1_out, a2_out = a2_out, a1_out
        applied.append("swap(data)")

    # --- resolve output branch names ---
    out_b1_default, out_b2_default = args.in_b1, args.in_b2
    if args.out_b1 is None and args.out_b2 is None:
        if "swapnames" in modes:
            out_b1_default, out_b2_default = out_b2_default, out_b1_default
            applied.append("swapnames(names)")
        elif "swap" in modes:
            out_b1_default, out_b2_default = out_b2_default, out_b1_default
            applied.append("names_follow_data")

    out_b1 = args.out_b1 if args.out_b1 else out_b1_default
    out_b2 = args.out_b2 if args.out_b2 else out_b2_default
    resolved_tree = tname if args.tree_name == "auto" else args.tree_name

    # Write output
    write_with_pyroot(args.out_path, resolved_tree, a1_out, a2_out, out_b1, out_b2)

    # Print summary
    print(f"[OK] Modes={args.mode} -> Applied={modes} ({', '.join(applied)}).")
    print(f"     Jets in: {len(a1)} → Jets out: {len(a1_out)}.")
    print(f"[INFO] Output branches: '{out_b1}', '{out_b2}' in tree '{resolved_tree}'.")
    if "top10" in modes:
        nlen = ak.to_numpy(ak.num(a1_out))
        assert np.all(nlen == 10)
        print("[INFO] All jets have exactly 10 emissions.")


if __name__ == "__main__":
    main()
