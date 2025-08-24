#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 14:41:55 2025

@author: ningyan
"""

# discretize_auto.py
import os
import numpy as np
import pandas as pd
import uproot
import awkward as ak
from argparse import ArgumentParser


def ensure_list(x, n_features):
    """Convert --nBins input into a list of length n_features"""
    if isinstance(x, int):
        return [x] * n_features
    if hasattr(x, "__len__"):
        if len(x) == 1:
            return [x[0]] * n_features
        if len(x) == n_features:
            return list(x)
    raise ValueError(f"--nBins must be either 1 or {n_features} integers, received {x}")


def to_dataframe(arr3d, feat_names):
    """
    arr3d: (N, L, F)
    feat_names: feature names, length = F
    Return a flattened DataFrame with column names like feat_0, feat_1, ...
    """
    N, L, F = arr3d.shape
    flat = arr3d.reshape(N, L * F)
    cols = []
    for i in range(L):
        for j, nm in enumerate(feat_names):
            cols.append(f"{nm}_{i}")
    return pd.DataFrame(flat, columns=cols)


def discretize_tree(
    tree,
    feat_names,
    out_h5,
    tag,
    nBins,
    num_const=200,
    pad_val=-1.0,
    lower_q=None,
    upper_q=None,
    auto_const_q=0.9
):
    """
    Discretize all feat_names branches in a TTree and write to HDF5.
    """
    # 1) Read as Awkward Array
    arrs = tree.arrays(feat_names, library="ak")

    # ----- Automatically choose sequence length based on quantile auto_const_q -----
    if (auto_const_q is not None) and (auto_const_q > 0) and (auto_const_q <= 1):
        # Use the length distribution of one branch (usually all have the same length)
        lens = ak.to_numpy(ak.num(arrs[feat_names[0]]))
        target_L = max(1, int(np.quantile(lens, auto_const_q)))
    else:
        target_L = num_const
    print(f"[len] target_L={target_L} (auto_const_q={auto_const_q})")
    # -------------------------------------------------------------------------------

    # 2) Normalize sequence length (truncate/pad)
    #    For each branch: pad/clip to target_L; None -> pad_val
    per_feat = []
    for name in feat_names:
        jag = arrs[name]
        padded = ak.pad_none(jag, target_L, clip=True)  # (N, target_L) with None, then replace with pad_val
        filled = ak.fill_none(padded, pad_val)
        per_feat.append(np.asarray(filled, dtype=np.float32))  # (N, L)

    # (N, L, F)
    stacked = np.stack(per_feat, axis=-1)
    N, L, F = stacked.shape

    # 3) Create/load bin edges for each feature
    bins_dir = "preprocessing_bins"
    os.makedirs(bins_dir, exist_ok=True)

    all_bins = []
    for fi, name in enumerate(feat_names):
        # Valid values: not equal to pad_val
        valid = stacked[..., fi][stacked[..., fi] != pad_val]
        if valid.size == 0:
            # If all are padding, create dummy bins to avoid crash
            valid = np.array([0.0, 1.0], dtype=np.float32)

        # Optional quantile clipping
        if lower_q is not None and upper_q is not None:
            lo = np.quantile(valid, lower_q)
            hi = np.quantile(valid, upper_q)
        else:
            lo, hi = np.min(valid), np.max(valid)
        if lo == hi:
            hi = lo + 1e-6

        #path = os.path.join(bins_dir, f"{name}_bins_{tag}.npy")
        path = os.path.join(bins_dir, f"var{fi+1}_bin.npy")
        if os.path.exists(path):
            bins = np.load(path)
            print(f"[bins] Loaded {path} (len={len(bins)})")
        else:
            bins = np.linspace(lo, hi, nBins[fi])
            np.save(path, bins)
            print(f"[bins] Saved  {path} (len={len(bins)})  range=({lo:.4g},{hi:.4g})")
        all_bins.append(bins)

    # 4) Discretization
    disc_feats = []
    for fi, bins in enumerate(all_bins):
        disc = np.digitize(stacked[..., fi], bins).astype(np.int32)
        disc = np.minimum(disc, len(bins) - 1)  # ★ Important: prevent out-of-bounds
        disc = disc.astype(np.int16)
        disc[stacked[..., fi] == pad_val] = -1
        disc_feats.append(disc)
    disc_stacked = np.stack(disc_feats, axis=-1)  # (N, L, F)  ← must keep

    '''
    # (old version for reference)
    disc_feats = []
    for fi, bins in enumerate(all_bins):
        disc = np.digitize(stacked[..., fi], bins).astype(np.int16)  # [0..len(bins)]
        # Mark padding positions as -1
        disc[stacked[..., fi] == pad_val] = -1
        disc_feats.append(disc)
    disc_stacked = np.stack(disc_feats, axis=-1)  # (N, L, F)
    '''

    # 5) Write HDF5: raw / discretized
    raw_df = to_dataframe(stacked, feat_names)
    disc_df = to_dataframe(disc_stacked, feat_names)

    raw_df.to_hdf(out_h5, key="raw", mode="w", complevel=9)
    disc_df.to_hdf(out_h5, key="discretized", mode="r+", complevel=9)

    print(f"[write] {out_h5}  raw={raw_df.shape}  disc={disc_df.shape}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", "-I", type=str, required=True,
                        help="Path to ROOT file (including filename)")
    parser.add_argument("--tag", type=str, required=True,
                        help="Tag used for binning/output naming, to distinguish different settings")
    parser.add_argument("--nBins", "-n", type=int, nargs="+", required=True,
                        help="Number of bins per feature (1 integer applies to all features, "
                             "or provide one integer per feature)")
    parser.add_argument("--num_const", type=int, default=200,
                        help="Maximum sequence length per event (truncate/pad)")
    parser.add_argument("--lower_q", type=float, default=None,
                        help="Optional: lower quantile clipping, e.g., 0.0")
    parser.add_argument("--upper_q", type=float, default=None,
                        help="Optional: upper quantile clipping, e.g., 1.0")
    parser.add_argument("--auto_const_q", type=float, default=0.9,
                        help="Automatically choose sequence length by quantile (0,1]; "
                             "<=0 disables this and uses --num_const instead")
    args = parser.parse_args()

    root_path = args.data_path
    assert os.path.isfile(root_path), f"File not found: {root_path}"

    # Output directory
    out_dir = os.path.join(os.path.dirname(root_path), "discretized")
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(root_path))[0]

    # Open ROOT file and enumerate all TTrees
    with uproot.open(root_path) as f:
        # Only keep TTrees
        tree_items = [(k, f[k]) for k in f.keys() if hasattr(f[k], "arrays")]

        if not tree_items:
            raise RuntimeError("No TTree found in this ROOT file.")

        for key, tree in tree_items:
            # Treename looks like 'lundtree;1', remove ';cycle'
            treename = key.split(";")[0]
            feat_names = [br.name for br in tree.branches]

            # Skip if no valid branches
            if len(feat_names) == 0:
                print(f"[skip] {treename} has no branches")
                continue

            # Expand nBins to match number of features
            nbins_vec = ensure_list(args.nBins, len(feat_names))
            out_h5 = os.path.join(out_dir, f"{base}_{treename}_{args.tag}.h5")

            print(f"\n[tree] {treename}  features={len(feat_names)} -> {out_h5}")
            print(f"       columns: {', '.join(feat_names[:8])}{' ...' if len(feat_names)>8 else ''}")

            discretize_tree(
                tree=tree,
                feat_names=feat_names,
                out_h5=out_h5,
                tag=args.tag,
                nBins=nbins_vec,
                num_const=args.num_const,
                lower_q=args.lower_q,
                upper_q=args.upper_q,
                auto_const_q=args.auto_const_q,
            )

            # Preview small sample
            try:
                with pd.HDFStore(out_h5, mode="r") as store:
                    for k in ["raw", "discretized"]:
                        if k in store:
                            df = store[k]
                            print(f"[preview] {k} head:\n{df.iloc[:5, :min(10, df.shape[1])]}\n")
            except Exception as e:
                print(f"[preview] Failed to read {out_h5}: {e}")


if __name__ == "__main__":
    main()