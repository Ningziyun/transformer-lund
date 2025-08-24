import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import time, os
import math
import ROOT 
from argparse import ArgumentParser
from array import array

torch.multiprocessing.set_sharing_strategy("file_system")


def set_seeds(seed):
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
parser.add_argument("--preprocessingBins", type=str, default="pt80_eta60_phi60_lower001")

args = parser.parse_args()

set_seeds(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

n_batches = args.num_samples // args.batchsize
rest = args.num_samples % args.batchsize

# Load model for sampling
model = torch.load(os.path.join(args.model_dir, args.model_name), weights_only=False)
model.classifier = False
model.to(device)
model.eval()

jets = []
bins = []
start = time.time()
for i in tqdm(range(n_batches), total=n_batches, desc="Sampling batch"):
    _jets, _bins = model.sample(
        starts=torch.zeros((args.batchsize, 3), device=device),
        device=device,
        len_seq=args.num_const + 1,
        trunc=args.trunc,
    )
    _jets = _jets.cpu().numpy()
    _bins = _bins.cpu().numpy()
    _jets_tmp = np.zeros((args.batchsize, args.num_const + 1, 3))
    _jets_tmp[:, :_jets.shape[1]] = _jets
    _bins_tmp = np.zeros((args.batchsize, args.num_const + 1))
    _bins_tmp[:, :_bins.shape[1]] = _bins
    jets.append(_jets_tmp)
    bins.append(_bins_tmp)

if rest != 0:
    _jets, _bins = model.sample(
        starts=torch.zeros((rest, 3), device=device),
        device=device,
        len_seq=args.num_const+1,
        trunc=args.trunc,
    )
    _jets = _jets.cpu().numpy()
    _bins = _bins.cpu().numpy()
    _jets_tmp = np.zeros((rest, args.num_const + 1, 3))
    _jets_tmp[:, :_jets.shape[1]] = _jets
    _bins_tmp = np.zeros((rest, args.num_const + 1))
    _bins_tmp[:, :_bins.shape[1]] = _bins
    jets.append(_jets_tmp)
    bins.append(_bins_tmp)

jets = np.concatenate(jets, 0)[:, 1:]
bins = np.concatenate(bins, 0)
dels = np.where(jets[:, 0, :].sum(-1) == 0)
bins = np.delete(bins, dels, axis=0)
jets = np.delete(jets, dels, axis=0)

print(f"Time needed {(time.time() - start) / float(len(jets))} seconds per jet")
print(f"\t{int(time.time() - start)} seconds in total for {len(jets)} jets")

file = ROOT.TFile.Open("transformerJets.root", "RECREATE")
tree = ROOT.TTree("tree","tree")

constit_pt = ROOT.std.vector[float]()
tree.Branch("constit_pt", constit_pt)
constit_eta = ROOT.std.vector[float]()
tree.Branch("constit_eta", constit_eta)
constit_phi = ROOT.std.vector[float]()
tree.Branch("constit_phi", constit_phi)

pt_bins = np.load(args.preprocessingDir+"/pt_bins_" + args.preprocessingBins + ".npy")
eta_bins = np.load(args.preprocessingDir+"/eta_bins_" + args.preprocessingBins + ".npy")
phi_bins = np.load(args.preprocessingDir+"/phi_bins_" + args.preprocessingBins + ".npy")

for jet in jets:
    constit_pt_binned = []
    constit_eta_binned = []
    constit_phi_binned = []

    # Clear the contents of the vector
    constit_pt.clear()
    constit_eta.clear()
    constit_phi.clear()
    # Replace the contents in the vector with the contents
    # from the current array
    constit_pt.reserve(len(jet))
    constit_eta.reserve(len(jet))
    constit_phi.reserve(len(jet))
    for constit in jet:
        constit_pt_binned.append(constit[0])
        constit_eta_binned.append(constit[1])
        constit_phi_binned.append(constit[2])

    mask = constit_pt_binned == 0

    constit_pt_tmp = (constit_pt_binned - np.random.uniform(0.0, 1.0, size=np.array(constit_pt_binned).shape)) * (
        pt_bins[1] - pt_bins[0]
    ) + pt_bins[0]
    constit_eta_tmp = (constit_eta_binned - np.random.uniform(0.0, 1.0, size=np.array(constit_eta_binned).shape)) * (
        eta_bins[1] - eta_bins[0]
    ) + eta_bins[0]
    constit_phi_tmp = (constit_phi_binned - np.random.uniform(0.0, 1.0, size=np.array(constit_phi_binned).shape)) * (
        phi_bins[1] - phi_bins[0]
    ) + phi_bins[0]

    # Probably this could be handled better, but it works fine for now
    for i in range(len(constit_pt_tmp)):
      constit_pt.push_back(math.exp(constit_pt_tmp[i]))
      constit_eta.push_back(constit_eta_tmp[i])
      constit_phi.push_back(constit_phi_tmp[i])

    tree.Fill()

file.WriteObject(tree, "tree")
file.Close()


jets[jets.sum(-1) == 0] = -1
n, c, f = np.shape(jets)
data = jets.reshape(n, c * f)
cols = [
    item
    for sublist in [f"PT_{i},Eta_{i},Phi_{i}".split(",") for i in range(c)]
    for item in sublist
]
df = pd.DataFrame(data, columns=cols)
df.to_hdf(os.path.join(args.model_dir, f"samples_{args.savetag}.h5"), key="discretized")

