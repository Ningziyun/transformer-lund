#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_jets_auto_discrete.py

Generate samples from a trained model using the SAME CLI and inputs as
`sample_jets_auto.py` but **do not** de-discretize / restore real values.
Instead, save the raw (discrete) bin indices directly to an HDF5 file in a
wide-table format analogous to `discretize_auto.py`:

  columns: Var1_0, Var2_0, ..., VarF_0, Var1_1, Var2_1, ..., VarF_1, ...

Padding uses -1 for all features at that time step, and padded steps are kept
in the output (so you can later mask them consistently).

Output:
  {model_dir}/samples_bins_{savetag}.h5  (HDF5, key="discretized")

Usage example (identical flags to sample_jets_auto.py):
  python sample_jets_auto_discrete.py \
      --model_dir models/test \
      --model_name model_last.pt \
      --savetag test \
      --num_samples 1000 \
      --batchsize 100 \
      --num_const 50 \
      --seed 0 \
      --trunc 0.95 \
      --preprocessingDir preprocessing_bins

Author: ChatGPT (per user's specification)
"""

import os
import time
import math
import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser

torch.multiprocessing.set_sharing_strategy("file_system")


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_parser():
    parser = ArgumentParser()
    # Keep flags in lockstep with sample_jets_auto.py
    parser.add_argument("--model_dir", type=str, default="models/test")
    parser.add_argument("--model_name", type=str, default="model_last.pt")
    parser.add_argument("--savetag", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--num_const", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trunc", type=float, default=None)
    parser.add_argument("--preprocessingDir", type=str, default="preprocessing_bins")
    # kept for backward-compatibility; not used here but present in sample_jets_auto.py
    parser.add_argument("--preprocessingBins", type=str, default="unused")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load model (saved with torch.save(model, ...)) ---
    model_path = os.path.join(args.model_dir, args.model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.load(model_path, weights_only=False)
    # Ensure generation mode
    if hasattr(model, "classifier"):
        model.classifier = False
    model.to(device)
    model.eval()

    # --- infer feature dimension F ---
    if hasattr(model, "num_features"):
        F = int(model.num_features)
    elif hasattr(model, "num_bins"):
        # some models carry per-feature bins list
        F = int(len(model.num_bins))
    else:
        # fall back to 3 features if not available
        F = 3

    n_batches = args.num_samples // args.batchsize
    rest = args.num_samples % args.batchsize

    # --- sampler: returns padded (bs, num_const+1, F) and (bs, num_const+1) ---
    @torch.no_grad()
    def sample_batch(bs: int):
        # Start tokens (one per feature).
        starts = torch.zeros((bs, F), device=device)
        # Model expected to implement .sample(starts, device, len_seq, trunc) -> (jets, bins)
        # jets: (bs, t, F) discrete (bin indices per feature per step), t <= num_const+1
        # bins: (bs, t)      discrete (bin-index summary per step), optional use downstream
        _jets, _bins = model.sample(starts=starts, device=device, len_seq=args.num_const + 1, trunc=args.trunc)
        _jets = _jets.detach().cpu().numpy()
        _bins = _bins.detach().cpu().numpy()

        # Pad time dimension to length num_const+1 using -1 (NOT 0, since 0 is a valid bin index)
        jets_pad = np.full((bs, args.num_const + 1, F), fill_value=-1, dtype=_jets.dtype)
        bins_pad = np.full((bs, args.num_const + 1),    fill_value=-1, dtype=_bins.dtype)
        t = _jets.shape[1]
        jets_pad[:, :t] = _jets
        bins_pad[:, :t] = _bins
        return jets_pad, bins_pad

    # --- sampling loop ---
    jets_chunks, bins_chunks = [], []
    start = time.time()
    for _ in range(n_batches):
        j, b = sample_batch(args.batchsize)
        jets_chunks.append(j)
        bins_chunks.append(b)
    if rest > 0:
        j, b = sample_batch(rest)
        jets_chunks.append(j)
        bins_chunks.append(b)

    jets = np.concatenate(jets_chunks, axis=0)[:, 1:]  # drop the start-token step
    bins = np.concatenate(bins_chunks, axis=0)

    # Drop events whose first real time step is fully padded (-1 for all features)
    dels = np.where((jets[:, 0, :] < 0).all(axis=-1))[0]
    if len(dels) > 0:
        jets = np.delete(jets, dels, axis=0)
        bins = np.delete(bins, dels, axis=0)

    print(f"Time per jet: {(time.time() - start) / float(max(1, len(jets))):.6f} s")
    print(f"Total time: {int(time.time() - start)} s for {len(jets)} jets")
    print(f"Feature dimension F = {F}")

    # IMPORTANT: We DO NOT restore real values using bin edges.
    # We save discrete bin indices directly in a wide HDF5 table, matching discretize_auto.py.

    # Make padded steps recognizable in downstream code by setting any padded time step to -1 across all features
    pad_steps = (jets < 0).any(axis=-1)
    jets[pad_steps] = -1

    # Shape to (N, C*F) with columns Var1_0, Var2_0, ..., VarF_0, Var1_1, ...
    n, c, f = jets.shape
    data = jets.reshape(n, c * f)
    cols = [f"Var{feat+1}_{i}" for i in range(c) for feat in range(f)]

    df = pd.DataFrame(data, columns=cols)
    out_h5 = os.path.join(args.model_dir, f"samples_bins_{args.savetag}.h5")
    os.makedirs(args.model_dir, exist_ok=True)
    df.to_hdf(out_h5, key="discretized")
    print(f"Wrote discretized samples to: {out_h5}")


if __name__ == "__main__":
    main()
