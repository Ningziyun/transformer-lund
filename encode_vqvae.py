#!/usr/bin/env python3

"""
Step 2: apply a trained VQ-VAE tokenizer to an input .h5 and write a codes .h5.

Output layout:
    codes                 (n_jets, n_const, n_levels) int32, -1 = padded constituent
    cond                  (n_jets, 5) float32  [pt, eta, phi, m, N_const]  (if input has a "jet" group)
    jet/{E,PX,PY,PZ}      copied from the input                           (if input has a "jet" group)
    attrs: num_codes, num_levels, n_const, frame, tokenizer, source

    python encode_jets.py --data train.h5 --tokenizer tokenizer.pt --out codes_train.h5
    python encode_jets.py --data test.h5  --tokenizer tokenizer.pt --out codes_test.h5
"""

import argparse
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import helpers
import helpers_vqvae
import models_vqvae

def encode(args):
    #load model
    device = torch.device(args.device)
    model, training_args = helpers_vqvae.load_checkpoint(args.tokenizer, device)

    # Load sata
    # no shuffling: batches arrive in file order, so they can be written to consecutive slices
    dataset = helpers_vqvae.constit_dataset(args.input, NConstituents=training_args["num_constituents"], add_mask=True, add_conditional=False, preprocess=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=training_args["num_workers"], pin_memory=device.type == "cuda", persistent_workers=False)
    N, C, Q = len(dataset), dataset.X.shape[1], model.Q #[Number jets, constituents, codebook]
    if C < training_args["num_constituents"]:
        print(f"note: file has only {C} constituent slots (requested {training_args['num_constituents']}), encoding {C}")

    #metrics for later
    t0, offset, n_real = time.time(), 0, 0
    usage = torch.zeros(Q, model.K, dtype=torch.long)
    sq_err, n_check = torch.zeros(model.F), 0

    #make the outfile
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        #Make the new output dataset
        dcodes = f.create_dataset("codes", shape=(N, C, Q), dtype=np.int32, compression="gzip", chunks=(min(N, 4096), C, Q))

        #Encode batch by batch, writing straight into the output file
        with torch.no_grad():
            for batch in loader:
                #Find the encoding
                X, mask, jet = helpers.format_input2(batch, store_mask=True, store_jet=False, device=device)
                codes = model.encode(X, mask)
                B = codes.shape[0]
                dcodes[offset:offset + B] = codes.cpu().numpy().astype(np.int32)

                #Count the usage
                real = ~mask
                n_real += int(real.sum())
                for q in range(Q):
                    usage[q] += torch.bincount(codes[..., q][real], minlength=model.K).cpu()

                # round-trip check (pre-processed space) on the first --check jets
                if n_check < args.check:
                    nb = min(B, args.check - n_check)
                    rec = model.decode(codes[:nb])
                    r = real[:nb]
                    sq_err += ((rec[r] - X[:nb][r].float()) ** 2).sum(0).cpu()
                    n_check += nb
                offset += B

        #Add jet dataset
        '''
        if has_jet:
            f["cond"] = dataset.jet_data
            with h5py.File(args.data, "r") as fin:
                g = f.create_group("jet")
                for k in helpers_vqvae.FOURVEC_KEYS:
                    g[k] = fin["jet"][k][:]
        '''
        #Add some attributes
        #f.attrs["num_codes"], f.attrs["num_levels"], f.attrs["n_const"], f.attrs["n_features"] = model.K, Q, N, model.F

    #Plot usage
    used = (usage > 0).sum(1).tolist()
    print(f"wrote {(N, C, Q)} codes ({n_real} real constituents, codes used per level {used}/{model.K}) "
          f"-> {args.out} [{time.time() - t0:.1f}s]")

    #Make debug plot
    if n_check:
        sd = torch.from_numpy(dataset.X[~dataset.mask].std(0)).clamp_min(1e-12)
        n_rc = int((~dataset.mask[:n_check]).sum())
        rmse = torch.sqrt(sq_err / max(n_rc, 1)) / sd
        #print(f"round-trip check on {n_check} jets: RMSE/std per feature " + ", ".join(f"{k} {v:.4f}" for k, v in zip(s["feature_keys"], rmse.tolist())))
        print(f"round-trip check on {n_check} jets: RMSE/std per feature " + ", ".join(f"{k} {v:.4f}" for k, v in enumerate(rmse.tolist())))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    helpers_vqvae.add_common_args(parser)
    parser.add_argument("--input", required=True, help="input .h5 (constituents/ and optionally jet/ groups)")
    parser.add_argument("--tokenizer", required=True, help="tokenizer checkpoint from train_vqvae.py")
    parser.add_argument("--out", required=True, help="output codes .h5")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--check", type=int, default=1000, help="jets to decode back as a sanity check (0 = skip)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    encode(args)
