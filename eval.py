import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model import JetTransformer
from preprocess import preprocess_dataframe

from argparse import ArgumentParser
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve

torch.multiprocessing.set_sharing_strategy('file_system')

def load_model(model_dir, name):
    model = torch.load(os.path.join(model_dir, f'model_{name}.pt'))
    return model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models/test', help="Model directory")
    parser.add_argument("--model_name", type=str, default='last', help="Checkpoint name")
    parser.add_argument("--data_path", type=str, default='inputFiles/top_benchmark/', help="Path to training data file")
    parser.add_argument("--data_split", type=str, default='test', help="Split to evaluate")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--num_const", type=int, default=100, help="Number of constituents")
    parser.add_argument("--num_events", type=int, default=200000, help="Number of events")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
    parser.add_argument("--bkg", type=str, default='qcd', choices=['qcd', 'top'])
    parser.add_argument("--reverse", action="store_true", help="Reverse the pt ordering")
    parser.add_argument("--limit_const", action="store_true", help="Only use jets with at least num_const constituents")
    parser.add_argument("--kind", type=str, default='perp', choices=['perp', 'log'], help='Kind of anomaly score')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on device {device}')

    if args.kind == 'perp':
        perp, log = True, False
    elif args.kind == 'log':
        perp, log = False, True

    num_features = 3
    num_bins = (41, 31, 31)

    print(f'Loading model from {args.model_dir+args.bkg}')
    model = load_model(args.model_dir+args.bkg, 'last')
    model.eval()
    model.to(device)

    scores = []
    labels = []
    nparts = []
    losses = []
    probs_min = []
    probs_min_idx = []
    probs_max = []
    probs_max_idx = []
    for c in ['qcd', 'top',]:
        print(c)
        tmp_losses = []
        data_path = os.path.join(args.data_path, f'test_{c}_30_bins.h5')
        df = pd.read_hdf(data_path, 'discretized', stop=args.num_events)
        x, padding_mask, bins = preprocess_dataframe(df,
            num_features=num_features,
            num_bins=num_bins,
            num_const=args.num_const,
            to_tensor=True,
            reverse=args.reverse,
            limit_nconst=args.limit_const
            )
        labels.append(int(c!=args.bkg) * np.ones(len(x)))

        test_dataset = TensorDataset(x, padding_mask, bins)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            )

        for x, padding_mask, true_bin in tqdm(test_loader,
                total=len(test_loader),
                desc=f'Evaluating {data_path}'
                ):
            nparts.append(padding_mask.sum(-1))
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            true_bin = true_bin.to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits = model(x, padding_mask)
                    probs = torch.nn.Softmax(dim=-1)(logits)
                    tmp = probs.min(-1)
                    prob_min = tmp.values.cpu().detach().numpy()
                    prob_min_idx = tmp.indices.cpu().detach().numpy()

                    tmp = probs.max(-1)
                    prob_max = tmp.values.cpu().detach().numpy()
                    prob_max_idx = tmp.indices.cpu().detach().numpy()

                    loss = model.loss_pC(logits, true_bin)
                    perplexity = model.probability(logits,
                        padding_mask,
                        true_bin,
                        perplexity=perp,
                        logarithmic=log
                        )
                    probs_min.append(prob_min)
                    probs_max.append(prob_max)
                    probs_min_idx.append(prob_min_idx)
                    probs_max_idx.append(prob_max_idx)
                    tmp_losses.append(loss.reshape(len(x), -1).cpu().detach().numpy())
                    scores.append(perplexity.cpu().detach().numpy())


        tmp_losses = np.concatenate(tmp_losses, axis=0)
        tmp_losses[bins[:, 1:] == -100] = np.nan
        losses.append(tmp_losses,)

    labels = np.concatenate(labels, axis=0)
    scores = np.concatenate(scores, axis=0)
    losses = np.concatenate(losses, axis=0)
    probs_min = np.concatenate(probs_min, axis=0)
    probs_min_idx = np.concatenate(probs_min_idx, axis=0)
    probs_max = np.concatenate(probs_max, axis=0)
    probs_max_idx = np.concatenate(probs_max_idx, axis=0)

    print(probs_min.shape)
    print(probs_min_idx.shape)

    probs = np.stack((probs_min, probs_max), axis=-1)
    probs_idx = np.stack((probs_min_idx, probs_max), axis=-1)
    print(probs.shape)
    print(probs_idx.shape)

    print(f'Loss {np.shape(losses)}')
    print(f'Label {np.shape(labels)}')
    print(f'Scores {np.shape(scores)}')
    print(f'Stats {np.shape(probs)}')

    print(f'Nparts {np.shape(nparts)}')

    np.savez(os.path.join(args.model_dir+args.bkg, f"predictions_{args.kind}{'_fixed' if args.limit_const else ''}.npz"),
        labels=labels,
        scores=scores,
        nparts=nparts,
        losses=losses,
        probs=probs,
        probs_idx=probs_idx,
        )
    auc = roc_auc_score(y_true=labels, y_score=-scores)
    print(auc)
