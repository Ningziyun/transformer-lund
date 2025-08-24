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
    # pad to fixed length (num_const+1) on time axis
    jets_pad = np.zeros((bs, args.num_const + 1, F), dtype=_jets.dtype)
    bins_pad = np.zeros((bs, args.num_const + 1), dtype=_bins.dtype)
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
dels = np.where(jets[:, 0, :].sum(-1) == 0)
bins = np.delete(bins, dels, axis=0)
jets = np.delete(jets, dels, axis=0)

print(f"Time per jet: {(time.time() - start) / float(len(jets)):.6f} s")
print(f"Total time: {int(time.time() - start)} s for {len(jets)} jets")
print(f"Feature dimension F = {F}")

# --- load per-feature bin edges: var1_bin.npy, var2_bin.npy, ... ---
bin_edges = []
for i in range(F):
    path = os.path.join(args.preprocessingDir, f"var{i+1}_bin.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing bin file: {path}")
    bin_edges.append(np.load(path))

# precompute uniform bin width (bins were saved with np.linspace)
bin_width = [float(b[1] - b[0]) for b in bin_edges]
bin_start = [float(b[0]) for b in bin_edges]

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

root_file.WriteObject(tree, "tree")
root_file.Close()

# --- also dump discretized jets to HDF5 (same as original script) ---
# mark padded time steps as -1 across all features
jets[jets.sum(-1) == 0] = -1
n, c, f = jets.shape  # n events, c constituents, f features
data = jets.reshape(n, c * f)

# column order: for each time step i -> Var1_i, Var2_i, ..., VarF_i
cols = [f"Var{feat+1}_{i}" for i in range(c) for feat in range(f)]

df = pd.DataFrame(data, columns=cols)
out_h5 = os.path.join(args.model_dir, f"samples_{args.savetag}.h5")
df.to_hdf(out_h5, key="discretized")
print(f"Wrote discretized samples to: {out_h5}")
