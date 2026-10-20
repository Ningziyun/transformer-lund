"""
Shared code for the VQ-VAE jet-constituent pipeline.

    jet_vq_common.py        data loading, kinematics, feature transform, VQ-VAE model (this file)
    train_vqvae.py          1. train the per-constituent VQ-VAE tokenizer
    encode_jets.py          2. encode an input .h5 into a codes .h5
    train_generate_gpt.py   3. train the next-token transformer on codes and generate jets

Input format (same as constit_dataset):
    f["constituents"]["E"|"PX"|"PY"|"PZ"]   shape (n_jets, n_max), padded entries E < 0
    f["jet"]["E"|"PX"|"PY"|"PZ"]            shape (n_jets,)   (optional, needed for
                                                               conditioning / relative jet)
Constituents are assumed to be stored contiguously (all real ones first, then padding),
in whatever order you want the transformer to learn (typically pT-ordered).
"""

import math

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import models_vqvae

# ---------------------------------------------------------------------
# Data loading / kinematics
# ---------------------------------------------------------------------
class constit_dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, NConstituents=20, add_mask=False, add_conditional=False, preprocess=None):
        super(constit_dataset, self).__init__()
        self.n_features=4
        self.pad_value=-10

        self.add_mask=add_mask
        self.preprocess=preprocess
        self.add_conditional=add_conditional

        with h5py.File(file_path, "r") as f:
            #Claude says its faster to stack and index, since slicing the h5py still loads everting in memory anyways
            const = f["constituents"]
            self.X = np.stack([np.asarray(const[k][:, :NConstituents], dtype=np.float32) for k in ["E","PX","PY","PZ"]], axis=-1)
            if self.add_conditional:
                j = f["jet"]
                jet4 = np.stack([np.asarray(j[k][:], dtype=np.float64) for k in ["E","PX","PY","PZ"]], axis=-1)

        #Mask which stores if padded. True/False=Padded/unpadded
        if add_mask or self.add_conditional:
          self.mask=np.all(self.X == np.float32(self.pad_value), axis=-1)
          self.n_real = int((~self.mask).sum())

        #Make the conditioning info once we know masks
        if self.add_conditional:
            E, px, py, pz = jet4.T
            pt = np.hypot(px, py)
            eta = np.arcsinh(pz / np.maximum(pt, 1e-12))
            phi = np.arctan2(py, px)
            m = np.sqrt(np.maximum(E ** 2 - (px ** 2 + py ** 2 + pz ** 2), 0.0))
            nconst = self.n_real
            self.jet_data = np.stack([pt, eta, phi, m, nconst], axis=-1).astype(np.float32)

    def __getitem__(self, index):
        X = torch.from_numpy(self.X[index])
        mask = torch.from_numpy(self.mask[index])
        if self.add_mask and self.add_conditional:
            return [X, torch.from_numpy(self.jet_data[index]), mask]
        elif self.add_mask:
            return [X, mask]
        elif self.add_conditional:
            return [X, torch.from_numpy(self.jet_data[index])]
        return X

    def __len__(self):
        return len(self.X)

def get_loaders(args):

    train_dataset = constit_dataset(args.train_file, NConstituents=args.num_constituents, add_mask=True, add_conditional=True, preprocess=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)
    test_dataset = constit_dataset(args.test_file, NConstituents=args.num_constituents, add_mask=True, add_conditional=True, preprocess=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)

    return train_loader,test_loader

def jet_PtEtaPhiM(jet):
    """(.., 4) E,px,py,pz -> pt, eta, phi, m, E"""
    E, px, py, pz = jet.unbind(-1)
    pt = torch.hypot(px, py)
    eta = torch.asinh(pz / pt.clamp_min(1e-8))
    phi = torch.atan2(py, px)
    p2 = px ** 2 + py ** 2 + pz ** 2
    m = torch.sqrt((E ** 2 - p2).clamp_min(0.0))
    return pt, eta, phi, m, E

def jet_conditional(jet, mask):
    """Same conditional vector as constit_dataset: [pt, eta, phi, m, N_const]"""
    pt, eta, phi, m, _ = jet_PtEtaPhiM(jet)
    n = (~mask).sum(-1).float()
    return torch.stack([pt, eta, phi, m, n], dim=-1)

def jet_from_cond(cond):
    """Jet frame (pt, eta, phi, E) from the conditional vector [pt, eta, phi, m, N]."""
    pt, eta, phi, m = cond[:, 0], cond[:, 1], cond[:, 2], cond[:, 3]
    E = torch.sqrt((pt * torch.cosh(eta)) ** 2 + m ** 2)
    return torch.stack([pt, eta, phi, E], dim=-1)

def frame_from_jet(jet):
    pt, eta, phi, _, E = jet_PtEtaPhiM(jet)
    return torch.stack([pt, eta, phi, E], dim=-1)

def jet_from_constituents(constit, mask):
    s = (constit * (~mask).unsqueeze(-1)).sum(1)
    return frame_from_jet(s)

def wrap_phi(phi):
    return torch.remainder(phi + math.pi, 2 * math.pi) - math.pi

# ---------------------------------------------------------------------
# Shared utilities for the scripts
# ---------------------------------------------------------------------
def format_input(data, input_format):
    return data["const"], data["jet"], data["mask"]

def load_tokenizer_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = models_vqvae.ConstituentVQVAE(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["train_args"]

def save_tokenizer_checkpoint(model, args, history, usage, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config,
                "state_dict": model.state_dict(),
                "n_const": args.num_constituents,
                "history": history,
                "reference": {"usage": usage},
                "train_args": vars(args)},
                path)

def write_fourvectors(group, constit):
    """Write (..., 4) array of (E, px, py, pz) as E/PX/PY/PZ datasets into an h5 group."""
    constit = constit.numpy() if torch.is_tensor(constit) else constit
    for k, name in enumerate(["E", "PX", "PY", "PZ"]):
        group[name] = constit[..., k]

def add_common_args(parser):
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",help="")
    parser.add_argument("--seed", type=int, default=0,help="")
