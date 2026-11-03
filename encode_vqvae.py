#!/usr/bin/env python3

"""
Apply a trained VQ-VAE tokenizer to an input .h5 and write a codes .h5.

Output layout:
    codes                 (n_jets, n_const, n_levels) int32, -1 = padded constituent
    cond                  (n_jets, 5) float32  [pt, eta, phi, m, N_const]  (if input has a "jet" group)
    jet/{E,PX,PY,PZ}      copied from the input                           (if input has a "jet" group)
    attrs: num_codes, num_levels, n_const, frame, tokenizer, source
"""

import argparse
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import helpers
import models_vqvae

def encode(args):

    # ---------------------------------------------------------------------
    # Preamble
    # ---------------------------------------------------------------------

    #load model
    device = torch.device(args.device)
    model, training_args = models_vqvae.load_tokenizer_checkpoint(args.tokenizer, device)

    # Load sata
    # no shuffling: batches arrive in file order, so they can be written to consecutive slices
    loader=helpers.get_loader(args.input, training_args)
    N, C, Q = len(loader.dataset), loader.dataset.X.shape[1], model.Q #[Number jets, constituents, codebook]
    if C < training_args["num_constituents"]:
        print(f"note: file has only {C} constituent slots (requested {training_args['num_constituents']}), encoding {C}")

    #metrics for later
    starttime, offset, n_real = time.time(), 0, 0
    usage = torch.zeros(Q, model.K, dtype=torch.long)
    sq_err, n_check = torch.zeros(model.F), 0

    # ---------------------------------------------------------------------
    # Make the output file
    # ---------------------------------------------------------------------
    #make the outfile
    with h5py.File(args.output, "w") as f:
        #Make the new output dataset
        grp_const = f.create_group("constituents")
        dset_codes = grp_const.create_dataset("codes", shape=(N, C, Q), dtype=np.int32, compression="gzip", chunks=(min(N, 4096), C, Q))
        grp_jet = f.create_group("jet")
        dset_jetPt = grp_jet.create_dataset("Pt", shape=(0,), maxshape=(None,), dtype="float32")
        dset_jetEta = grp_jet.create_dataset("Eta", shape=(0,), maxshape=(None,), dtype="float32")
        dset_jetPhi = grp_jet.create_dataset("Phi", shape=(0,), maxshape=(None,), dtype="float32")
        dset_jetM = grp_jet.create_dataset("M", shape=(0,), maxshape=(None,), dtype="float32")

        #Encode batch by batch, writing straight into the output file
        with torch.no_grad():
            for batch in loader:
                #Find the encoding
                X, mask, jet = helpers.format_input(batch, unpack_mask=True, unpack_jet=True, device=device)
                codes = model.encode(X, mask)
                B = codes.shape[0]
                dset_codes[offset:offset + B] = codes.cpu().numpy().astype(np.int32)

                new_size = dset_jetPt.shape[0] + B
                dset_jetPt.resize(new_size, axis=0)
                dset_jetEta.resize(new_size, axis=0)
                dset_jetPhi.resize(new_size, axis=0)
                dset_jetM.resize(new_size, axis=0)
                dset_jetPt[offset:offset + B]=jet[...,0].cpu().numpy().astype(np.float32).flatten()
                dset_jetEta[offset:offset + B]=jet[...,1].cpu().numpy().astype(np.float32).flatten()
                dset_jetPhi[offset:offset + B]=jet[...,2].cpu().numpy().astype(np.float32).flatten()
                dset_jetM[offset:offset + B]=jet[...,3].cpu().numpy().astype(np.float32).flatten()

                #Count the usage
                real = ~mask
                n_real += int(real.sum())
                for q in range(Q):
                    usage[q] += torch.bincount(codes[..., q][real], minlength=model.K).cpu()

                #On first args.check constituents caluclate the MSE
                if n_check < args.check:
                    nb = min(B, args.check - n_check) #how much of batch to take
                    mini_mask = real[:nb]

                    #Get the MSE
                    recon = model.decode(codes[:nb])
                    sq_err += ((recon[mini_mask] - X[:nb][mini_mask].float()) ** 2).sum(0).cpu()
                    n_check += nb
                offset += B

        #Add some attributes
        #f.attrs["num_codes"], f.attrs["num_levels"], f.attrs["n_const"], f.attrs["n_features"] = model.K, Q, N, model.F

    # ---------------------------------------------------------------------
    # Debug info
    # ---------------------------------------------------------------------
    #Plot usage
    codes_used = (usage > 0).sum(1).tolist()
    print(f"wrote {(N, C, Q)} codes ({n_real} real constituents, codes used per level {codes_used}/{model.K}) "
          f"-> {args.output} [{time.time() - starttime:.1f}s]")

    #Make the reco-RMSE vs feature std debug info
    if n_check:
        std = torch.from_numpy(loader.dataset.X[~loader.dataset.mask].std(0)).clamp_min(1e-12)
        n_rc = int((~loader.dataset.mask[:n_check]).sum())
        RMSE = torch.sqrt(sq_err / max(n_rc, 1)) / std 
        print(f"round-trip check on {n_check} jets: RMSE/std per feature " + ", ".join(f"{k} {v:.4f}" for k, v in enumerate(RMSE.tolist())))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="input .h5 (constituents/ and optionally jet/ groups)")
    parser.add_argument("--output", required=True, help="output codes .h5")
    parser.add_argument("--tokenizer", required=True, help="tokenizer checkpoint from train_vqvae.py")
    parser.add_argument("--store-jet", action="store_true", help="Store jet info")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--check", type=int, default=1000, help="jets to decode back as a sanity check (0 = skip)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",help="")
    parser.add_argument("--seed", type=int, default=0,help="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    encode(args)
