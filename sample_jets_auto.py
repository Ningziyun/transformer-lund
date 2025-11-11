import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import time, os
import math
import ROOT
from argparse import ArgumentParser

torch.multiprocessing.set_sharing_strategy("file_system")


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


parser = ArgumentParser()
parser.add_argument("--model_dir", type=str, default="models/test")
parser.add_argument("--model_name", type=str, default="model_last.pt")
parser.add_argument("--savetag", type=str, default="test")
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--batchsize", type=int, default=100)
parser.add_argument("--num_const", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--trunc", type=float, default=None)
parser.add_argument("--preprocessingDir", type=str, default="preprocessing_bins")
# kept for backward-compatibility but not used in the new naming
parser.add_argument("--preprocessingBins", type=str, default="unused")
#---Step-14---
# --- added: optional manual prefix for preprocessing_bins ---
parser.add_argument("--bin_prefix", type=str, default=None,
                    help="Optional prefix for preprocessing_bins (e.g. 'log_kt_deltaR'). "
                         "If not provided, automatically inferred from arguments.txt or fallback to var{i}_bin.npy.")
#---End-Step-14---

args = parser.parse_args()
set_seeds(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- load model ---
model_path = os.path.join(args.model_dir, args.model_name)
model = torch.load(model_path, weights_only=False)
model.classifier = False
model.to(device)
model.eval()

# infer feature dimension F
if hasattr(model, "num_features"):
    F = int(model.num_features)
elif hasattr(model, "num_bins"):
    F = int(len(model.num_bins))
else:
    F = 3  # last-resort fallback

n_batches = args.num_samples // args.batchsize
rest = args.num_samples % args.batchsize

# --- helper to sample one batch with shape adapted to F ---
def sample_batch(bs: int):
    starts = torch.zeros((bs, F), device=device)  # start token per feature
    _jets, _bins = model.sample(
        starts=starts, device=device, len_seq=args.num_const + 1, trunc=args.trunc
    )
    _jets = _jets.cpu().numpy()        # (bs, t, F) with t<=num_const+1
    _bins = _bins.cpu().numpy()        # (bs, t)
    '''
    Step-7
    # pad to fixed length (num_const+1) on time axis
    jets_pad = np.zeros((bs, args.num_const + 1, F), dtype=_jets.dtype)
    bins_pad = np.zeros((bs, args.num_const + 1), dtype=_bins.dtype)
    '''
    # Step-7 Adding
    # Use -1 as the padding value so that bin index 0 remains a valid class.
    jets_pad = np.full((bs, args.num_const + 1, F), fill_value=-1, dtype=_jets.dtype)
    bins_pad = np.full((bs, args.num_const + 1), fill_value=-1, dtype=_bins.dtype)
    # 0 is a valid bin index in 0-based coding; using -1 avoids confusing padding with a real bin.
    # Step-7 Ending
    jets_pad[:, : _jets.shape[1]] = _jets
    bins_pad[:, : _bins.shape[1]] = _bins
    return jets_pad, bins_pad


# --- sampling loop ---
jets_chunks, bins_chunks = [], []
start = time.time()
for _ in tqdm(range(n_batches), total=n_batches, desc="Sampling batch"):
    j, b = sample_batch(args.batchsize)
    jets_chunks.append(j)
    bins_chunks.append(b)
if rest > 0:
    j, b = sample_batch(rest)
    jets_chunks.append(j)
    bins_chunks.append(b)

jets = np.concatenate(jets_chunks, axis=0)[:, 1:]  # drop start token step
bins = np.concatenate(bins_chunks, axis=0)

# drop jets whose first time-step is all zeros (typical "empty" ones)
'''
Step-7 Deleted
dels = np.where(jets[:, 0, :].sum(-1) == 0)
'''
# Step-7 Adding
# Drop events whose first time step is fully padded (-1 across all features).
dels = np.where((jets[:, 0, :] < 0).all(axis=-1))
# Step-7 Ending
bins = np.delete(bins, dels, axis=0)
jets = np.delete(jets, dels, axis=0)

print(f"Time per jet: {(time.time() - start) / float(len(jets)):.6f} s")
print(f"Total time: {int(time.time() - start)} s for {len(jets)} jets")
print(f"Feature dimension F = {F}")

'''
Step-14 Deleted
# --- load per-feature bin edges: var1_bin.npy, var2_bin.npy, ... ---
bin_edges = []
for i in range(F):
    path = os.path.join(args.preprocessingDir, f"var{i+1}_bin.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing bin file: {path}")
    bin_edges.append(np.load(path))
'''
#---Step-14---
# --- determine bin prefix before loading ---
# Before:
# bin_edges = []
# for i in range(F):
#     path = os.path.join(args.preprocessingDir, f"var{i+1}_bin.npy")
#     if not os.path.isfile(path):
#         raise FileNotFoundError(f"Missing bin file: {path}")
#     bin_edges.append(np.load(path))
# After:
bin_prefix = None

# 1. manual input mode (highest priority)
if args.bin_prefix:
    bin_prefix = args.bin_prefix
    print(f"[bin_prefix] Using manual prefix: {bin_prefix}")

# 2. automatic mode: read arguments.txt under model_dir
else:
    arg_path = os.path.join(args.model_dir, "arguments.txt")
    if os.path.isfile(arg_path):
        try:
            with open(arg_path, "r") as f:
                for line in f:
                    if line.strip().startswith("data_path"):
                        # example line: data_path            inputFiles/log_kt_deltaR_train.h5
                        val = line.strip().split()[-1]
                        base = os.path.splitext(os.path.basename(val))[0]
                        bin_prefix = base.replace("_train", "").replace("_val", "")
                        print(f"[bin_prefix] Auto-detected from arguments.txt: {bin_prefix}")
                        break
        except Exception as e:
            print(f"[bin_prefix] Failed to read arguments.txt: {e}")

# 3. fallback: no prefix
if bin_prefix is None:
    print("[bin_prefix] No prefix detected, falling back to var{i}_bin.npy mode.")

# --- load bin edges ---
bin_edges = []
for i in range(F):
    if bin_prefix:
        path = os.path.join(args.preprocessingDir, f"{bin_prefix}_var{i+1}_bin.npy")
    else:
        path = os.path.join(args.preprocessingDir, f"var{i+1}_bin.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing bin file: {path}")
    bin_edges.append(np.load(path))
#---Step-14-End---
'''
Step-6 quantile Deleted
# precompute uniform bin width (bins were saved with np.linspace)
bin_width = [float(b[1] - b[0]) for b in bin_edges]
bin_start = [float(b[0]) for b in bin_edges]
# Quantile bins are NOT uniform. There is no single width/start per feature.
# Remove the uniform-width precomputation and use per-index left/right edges instead.
'''

# --- write ROOT file with dynamic branches var1..varF ---
root_file = ROOT.TFile.Open("transformerJets.root", "RECREATE")
tree = ROOT.TTree("tree", "tree")

feat_vecs = [ROOT.std.vector[float]() for _ in range(F)]
for i in range(F):
    tree.Branch(f"var{i+1}", feat_vecs[i])

# unbin and fill
for jet in jets:  # jet: (T, F) discrete indices (may include -1 pads)
    T = jet.shape[0]

    # clear and reserve vectors
    for v in feat_vecs:
        v.clear()
        v.reserve(T)
    '''
    Step-4 Deleted
    # jitter uniformly inside the bin for each feature/time step
    rnd = np.random.uniform(0.0, 1.0, size=jet.shape)
    cont = np.empty_like(jet, dtype=np.float64)
    for fidx in range(F):
        cont[:, fidx] = (jet[:, fidx] - rnd[:, fidx]) * bin_width[fidx] + bin_start[fidx]

    # DO NOT apply exp to the first feature (and to none of the others)
    # push values to ROOT vectors
    for t in range(T):
        for fidx in range(F):
            feat_vecs[fidx].push_back(float(cont[t, fidx]))
    tree.Fill()
    '''
    # --- filter out padding steps before unbinning ---
    # Padding is encoded as 0 or -1 in the discrete indices. Those should NOT be unbinned.
    valid_mask = (jet >= 0).all(axis=1)   # keep only time steps where every feature has a positive bin index
    jet_valid = jet[valid_mask]
    T_valid = jet_valid.shape[0]

    # Nothing valid to write for this jet
    if T_valid == 0:
        continue

    # --- clamp indices to [1, nBins] just in case any stray values sneak in ---
    for fidx in range(F):
        n_bins = len(bin_edges[fidx]) - 1  # because bin_edges are boundaries of length nBins+1 *if* you ever switch
        # With current files, len(bin_edges[fidx]) is the number of boundaries used by np.digitize (k in [0 .. len(b)-1 or len(b)])
        # We still clamp to be safe:
        '''
        Step-7 Deleted
        jet_valid[:, fidx] = np.clip(jet_valid[:, fidx], 1, len(bin_edges[fidx]) - 1)
        '''
        # Step-7 Adding
        # Clamp to [0, nBins-1]; edges has length (nBins+1).
        jet_valid[:, fidx] = np.clip(jet_valid[:, fidx], 0, len(bin_edges[fidx]) - 2)
        # Step-7 Ending


    '''
    step-6 quantile deleted
    # --- unbin using 1-based indices: k ∈ {1..nBins} maps to [start + (k-1)*w, start + k*w) ---
    rnd = np.random.uniform(0.0, 1.0, size=jet_valid.shape)
    cont = np.empty_like(jet_valid, dtype=np.float64)
    for fidx in range(F):
        # convert discrete index to continuous by sampling uniformly within the bin
        cont[:, fidx] = bin_start[fidx] + (jet_valid[:, fidx] - 1 + rnd[:, fidx]) * bin_width[fidx]
    # Each discrete index k ∈ {1..nBins} corresponds to the interval [edges[k-1], edges[k]).
    # This works for quantile bins where widths vary across k.

    '''
    # Step-6 Adding
    # --- unbin with per-bin left/right edges (quantile bins are NOT uniform) ---
    rnd = np.random.uniform(0.0, 1.0, size=jet_valid.shape)
    cont = np.empty_like(jet_valid, dtype=np.float64)
    for fidx in range(F):
        '''
        Step-7 Deleted
        k = jet_valid[:, fidx].astype(np.int64)   # discrete indices in [1 .. nBins]
        edges = bin_edges[fidx]                   # shape: (nBins + 1,)
        # left/right edges for each individual index
        left  = edges[k - 1]
        right = edges[k]
        '''
        # Step-7 Adding
        # 0-based bins: each k ∈ {0..nBins-1} maps to interval [edges[k], edges[k+1]).
        k = jet_valid[:, fidx].astype(np.int64)
        edges = bin_edges[fidx]                   # shape: (nBins + 1,)
        left  = edges[k]
        right = edges[k + 1]
        # Step-7 Ending
        width = right - left
        # sample uniformly within the corresponding bin interval
        cont[:, fidx] = left + rnd[:, fidx] * width
    # Step-6 End


    # --- push only valid steps to ROOT vectors ---
    for v in feat_vecs:
        v.clear()
        v.reserve(T_valid)

    for t in range(T_valid):
        for fidx in range(F):
            feat_vecs[fidx].push_back(float(cont[t, fidx]))
    tree.Fill()


root_file.WriteObject(tree, "tree")
root_file.Close()

'''
Step-7 Deleted
# --- also dump discretized jets to HDF5 (same as original script) ---
# mark padded time steps as -1 across all features
jets[jets.sum(-1) == 0] = -1
'''
# Step-7 Adding
# Use negative entries to detect padded time steps (we use -1 as padding everywhere).
# Do NOT rely on 'sum == 0' because 0 is a valid bin index in 0-based coding.
pad_steps = (jets < 0).any(axis=-1)
jets[pad_steps] = -1
# Step-7 Ending
n, c, f = jets.shape  # n events, c constituents, f features
data = jets.reshape(n, c * f)

# column order: for each time step i -> Var1_i, Var2_i, ..., VarF_i
cols = [f"Var{feat+1}_{i}" for i in range(c) for feat in range(f)]

df = pd.DataFrame(data, columns=cols)
out_h5 = os.path.join(args.model_dir, f"samples_{args.savetag}.h5")
df.to_hdf(out_h5, key="discretized")
print(f"Wrote discretized samples to: {out_h5}")
