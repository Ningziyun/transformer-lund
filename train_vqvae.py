#!/usr/bin/env python3
"""
Step 1: train the per-constituent VQ-VAE tokenizer.

Each constituent's (already pre-processed) feature vector is encoded by an MLP, quantised with
a residual VQ (--levels codebooks of --codebook-size codes = --levels tokens per constituent)
and decoded back. Only valid constituents enter the loss; padded slots are identified with the
pad value of the pre-processing used to make the file (--preprocess, or --pad-value).

The checkpoint (--out) is rewritten after every epoch and stores the model config, the
per-epoch history and the validation code usage (reference for later shift checks).

    python train_vqvae.py --train-file train.h5 --test-file test.h5 --input_format 4vec \
                          --preprocess logstd --out tokenizer.pt [--levels 2 --codebook-size 512]
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

import helpers
import helpers_vqvae

import models_vqvae

import vq_diagnostics

def train(args):
    device = torch.device(args.device)

    #Load the data
    train_loader, test_loader = helpers_vqvae.get_loaders(args)
    print("train:", len(train_loader.dataset))
    print("test: ", len(test_loader.dataset))

    #Load the model
    model = models_vqvae.ConstituentVQVAE(args.codebook_size, args.levels, args.code_dim, args.hidden, args.input_format, n_features=train_loader.dataset.n_features, pad_value=train_loader.dataset.pad_value).to(device)

    #Set the optimizer and scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    N_batches = len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * N_batches)

    #Store some info
    history = []
    K, Q = args.codebook_size, args.levels

    #Loop over epochs
    for epoch in range(args.epochs):
        starttime = time.time()

        # ---------------- training ----------------
        model.train()
        tot_train_reco, tot_train_com, n_train = 0.0, 0.0, 0
        for batch, X in enumerate(train_loader):

            #input data; mask is True for PADDED slots, so valid constituents are X[~mask] -> (M, F)
            X, mask, _ = helpers.format_input2(X, store_mask=True, store_jet=False, device=device)
            valid = X[~mask]
            if valid.shape[0] == 0:
                continue

            #Calculate loss
            _, _, reco, commit, _ = model(valid)
            loss = reco + commit
            opt.zero_grad(set_to_none=True)

            #Backprop
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            #Step the optimizer
            opt.step()
            sched.step()

            #Constituent-weighted running averages (batches hold different numbers of constituents)
            w = valid.shape[0]
            tot_train_reco += reco.item() * w
            tot_train_com += commit.item() * w
            n_train += w

        if helpers.model_has_nonfinite_parameters(model):
            raise RuntimeError(f"non-finite model parameters after epoch {epoch + 1}; "
                               f"lower --lr or check the input features for inf/nan")

        # ---------------- validation (test file) ----------------
        model.eval()
        tot_test_reco, tot_test_com, n_test = 0.0, 0.0, 0
        counts = torch.zeros(Q, K, device=device)  # code usage over the WHOLE validation set
        with torch.no_grad():
            for batch, X in enumerate(test_loader):

                #input data and caluclate encoding
                X, mask, _ = helpers.format_input2(X, store_mask=True, store_jet=False, device=device)
                valid = X[~mask]
                if valid.shape[0] == 0:
                    continue
                _, codes, reco, commit, _ = model(valid)

                #Constituent-weighted running averages
                w = valid.shape[0]
                tot_test_reco += reco.item() * w
                tot_test_com += commit.item() * w
                n_test += w

                #Count number of code-books
                for q in range(Q):
                    counts[q] += torch.bincount(codes[:, q], minlength=K)

        #Add some other metrics from the validation
        probs = counts / counts.sum(1, keepdim=True).clamp_min(1)
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum(1)).tolist()
        used = (counts > 0).sum(1).tolist()

        #Training summary
        history.append({"epoch": epoch + 1,
                        "train_reco": tot_train_reco / max(n_train, 1),
                        "test_reco": tot_test_reco / max(n_test, 1),
                        "train_commit": tot_train_com / max(n_train, 1),
                        "test_commit": tot_test_com / max(n_test, 1),
                        "perplexity": perplexity,
                        "used": used,
                        "K": K,
                        "lr": sched.get_last_lr()[0],
                        })
        h = history[-1]
        print(f"epoch {epoch + 1:3d} | train reco {h['train_reco']:.5f} | test reco {h['test_reco']:.5f} "
              f"| commit {h['test_commit']:.5f} | perplexity {[round(p, 1) for p in perplexity]} "
              f"| codes used {used}/{K} | {time.time() - starttime:.1f}s", flush=True)

        # save every epoch so a killed batch job keeps its progress
        save_checkpoint(model, args, history, args.out)
        print(f"saved tokenizer -> {args.out}")

    # diagnostics on the test file; the full reference (code usage + quantisation-error quantiles,
    # used by later shift / OOD checks) is stored in the checkpoint
    col = vq_diagnostics.collect(model, test_loader.dataset.X, test_loader.dataset.mask, n_jets=args.diag_jets, device=args.device)
    print(f"saved tokenizer -> {args.out}")
 
    diag_dir = args.diag_dir or str(Path(args.out).with_suffix("")) + "_diag"
    vq_diagnostics.run_diagnostics(col, diag_dir, input_format=args.input_format, preprocess=args.preprocess,
                                   feature_keys=args.feature_keys, history=history, plots=not args.no_plots,
                                   projections=args.projections, max_points=args.max_points,
                                   )

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    helpers_vqvae.add_common_args(parser)

    #data / io
    parser.add_argument("--train-file", required=True, help="Path to training .h5")
    parser.add_argument("--test-file", required=True, help="Path to testing .h5")
    parser.add_argument("--input_format", type=str, choices=["ktdr", "4vec", "relvec"], default="ktdr", help="What format of inputs we are using")
    parser.add_argument("--feature-keys", nargs="+", default=list(helpers_vqvae.FOURVEC_KEYS), help="datasets under constituents/ to use as features (default: E PX PY PZ)")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size in JETS (each batch holds ~batch-size x <N_const> constituents)")
    parser.add_argument("--num-workers", type=int, default=1, help="DataLoader workers")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle training loader (default: True)")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable shuffle")
    parser.add_argument("--preprocess", type=str, default=None, choices=["log", "linear", "logit", "standardize", "logstd", "shiftpad"], help="Pre-processing ALREADY applied to the input files (not applied here); sets the pad value")
    parser.add_argument("--pad-value", type=float, default=None, help="Override the pad value (default: from --preprocess / --input_format)")
    parser.add_argument("--num-constituents", type=int, default=20, help="Number of constituents")

    parser.add_argument("--out", default="tokenizer.pt")
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--levels", type=int, default=2, help="residual VQ depth = tokens per constituent")
    parser.add_argument("--code-dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    d = parser.add_argument_group("diagnostics")
    d.add_argument("--diag-dir", help="where to write plots + metrics.json (default <out>_diag)")
    d.add_argument("--diag-jets", type=int, default=10000, help="max validation jets used for diagnostics")
    d.add_argument("--no-plots", action="store_true", help="compute and print metrics only")
    d.add_argument("--projections", nargs="*", default=["pca", "tsne"], choices=["pca", "tsne", "umap"])
    d.add_argument("--max-points", type=int, default=5000, help="constituents in the latent projections")
    args = parser.parse_args()

    helpers.set_seeds(args.seed)

    train(args)

if __name__ == "__main__":
    main()
