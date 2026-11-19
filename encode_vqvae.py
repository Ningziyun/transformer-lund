#!/usr/bin/env python3

"""
Apply a trained VQ-VAE tokenizer to an input .h5 and write a codes .h5.

The input is streamed in file order, so row i of the output is jet i of the input.

Output layout:
    constituents/codes    (n_jets, n_const, n_levels) int32, -1 = padded constituent
    cond                  (n_jets, 5) float32  [pt, eta, phi, m, N_const], the conditioning used by train_binned.py
    jet/{Pt,Eta,Phi,M}    (n_jets,) float32, the same jet kinematics as separate columns
    attrs: num_codes, num_levels, n_const, n_features, input_format, tokenizer, source
"""

import argparse
import time

import h5py
import numpy as np
import torch

import helpers
import models_vqvae

def encode(args):

    # ---------------------------------------------------------------------
    # Preamble
    # ---------------------------------------------------------------------

    #load model
    device = torch.device(args.device)
    model, training_args = models_vqvae.load_tokenizer_checkpoint(args.tokenizer, device)

    # Load data in FILE ORDER: no shuffling, one reader worker, keep the last partial batch, and never inherit a --max-jets the tokenizer may have been trained with
    loader_args = dict(training_args, shuffle=False, batch_size=args.batch_size, num_workers=min(args.num_workers, 1), max_jets=args.max_jets, device=args.device)
    loader = helpers.get_loader(args.input, loader_args, train=False)
    dataset = loader.dataset
    N, C, Q = dataset.n_jets, dataset.width, model.Q #[Number jets, constituents, codebook levels]
    if C < training_args["num_constituents"]:
        print(f"note: file has only {C} constituent slots (requested {training_args['num_constituents']}), encoding {C}")

    #metrics for later
    starttime, offset, n_real = time.time(), 0, 0
    usage = torch.zeros(Q, model.K, dtype=torch.long)
    sq_err, n_check, n_rc = torch.zeros(model.F, dtype=torch.float64), 0, 0
    feat_sum, feat_sumsq, feat_n = torch.zeros(model.F, dtype=torch.float64), torch.zeros(model.F, dtype=torch.float64), 0

    # ---------------------------------------------------------------------
    # Make the output file
    # ---------------------------------------------------------------------
    with h5py.File(args.output, "w") as f:
        #Make the new output datasets, sized up front (rows are written in file order)
        grp_const = f.create_group("constituents")
        dset_codes = grp_const.create_dataset("codes", shape=(N, C, Q), dtype=np.int32, compression="gzip", chunks=(max(1, min(N, 4096)), C, Q))
        dset_cond = f.create_dataset("cond", shape=(N, 5), dtype=np.float32)
        grp_jet = f.create_group("jet")
        dset_jet = {k: grp_jet.create_dataset(k, shape=(N,), dtype=np.float32) for k in ["Pt","Eta","Phi","M"]}

        #Encode batch by batch, writing straight into the output file
        with torch.no_grad():
            for batch in loader:
                #Find the encoding
                X, mask, jet = helpers.format_input(batch, unpack_mask=True, unpack_jet=True, device=device)
                codes = model.encode(X, mask)
                B = codes.shape[0]
                dset_codes[offset:offset + B] = codes.cpu().numpy().astype(np.int32)

                #Conditioning [pt, eta, phi, m, N_const], the same vector the original-format loader gives
                jet_np = jet.cpu().numpy().astype(np.float32)
                dset_cond[offset:offset + B] = jet_np
                for i, k in enumerate(["Pt","Eta","Phi","M"]):
                    dset_jet[k][offset:offset + B] = jet_np[:, i]

                #Count the usage, and the feature spread for the round-trip check
                real = ~mask
                n_real += int(real.sum())
                for q in range(Q):
                    usage[q] += torch.bincount(codes[..., q][real], minlength=model.K).cpu()
                Xr = X[real].double()
                feat_sum += Xr.sum(0).cpu()
                feat_sumsq += (Xr ** 2).sum(0).cpu()
                feat_n += Xr.shape[0]

                #On the first args.check jets calculate the MSE of decode(encode(x))
                if n_check < args.check:
                    nb = min(B, args.check - n_check) #how much of batch to take
                    mini_mask = real[:nb]
                    recon = model.decode(codes[:nb])
                    sq_err += ((recon[mini_mask] - X[:nb][mini_mask].float()) ** 2).sum(0).double().cpu()
                    n_rc += int(mini_mask.sum())
                    n_check += nb
                offset += B

        if offset != N:
            raise RuntimeError(f"wrote {offset} jets but the input has {N}; the codes file is incomplete")

        #Add some attributes
        f.attrs["num_codes"], f.attrs["num_levels"], f.attrs["n_const"], f.attrs["n_features"] = model.K, Q, C, model.F
        f.attrs["input_format"] = str(training_args["input_format"])
        f.attrs["tokenizer"], f.attrs["source"] = str(args.tokenizer), str(args.input)

    # ---------------------------------------------------------------------
    # Debug info
    # ---------------------------------------------------------------------
    #Plot usage
    codes_used = (usage > 0).sum(1).tolist()
    print(f"wrote {(N, C, Q)} codes ({n_real} real constituents, codes used per level {codes_used}/{model.K}) "
          f"-> {args.output} [{time.time() - starttime:.1f}s]")

    #Make the reco-RMSE vs feature std debug info (std over all real constituents of the file, accumulated above)
    if n_check:
        mean = feat_sum / max(feat_n, 1)
        std = torch.sqrt((feat_sumsq / max(feat_n, 1) - mean ** 2).clamp_min(0)).clamp_min(1e-12)
        RMSE = torch.sqrt(sq_err / max(n_rc, 1)) / std
        print(f"round-trip check on {n_check} jets: RMSE/std per feature " + ", ".join(f"{k} {v:.4f}" for k, v in enumerate(RMSE.tolist())))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="input .h5 (constituents/ and optionally jet/ groups)")
    parser.add_argument("--output", required=True, help="output codes .h5")
    parser.add_argument("--tokenizer", required=True, help="tokenizer checkpoint from train_vqvae.py")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=1, help="reader workers, at most 1 so the output keeps the input order")
    parser.add_argument("--max-jets", type=int, default=None, help="encode only the first N jets (quick tests)")
    parser.add_argument("--check", type=int, default=1000, help="jets to decode back as a sanity check (0 = skip)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",help="")
    parser.add_argument("--seed", type=int, default=0,help="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    encode(args)
