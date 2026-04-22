import torch, os
from argparse import ArgumentParser
from helpers_train import set_seeds, load_data
import numpy as np
from tqdm import tqdm


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model")
    parser.add_argument("--data", type=str, help="Path to data")
    parser.add_argument("--tag", type=str, help="Tag for storing results")
    parser.add_argument(
        "--num_const",
        type=int,
        default=50,
        help="Number of constituents per jet. Default 50",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for torch and numpy. Default 0",
    )
    parser.add_argument(
        "--num_events",
        type=int,
        default=None,
        help="Number of jets used. Defaults to all available in 'data'",
    )

    return parser.parse_args()


def get_probs(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)
    model.eval()
    probs = []
    n_const = []
    for x, mask, bins in tqdm(loader, total=len(loader)):
        with torch.no_grad():
            x, mask, bins = x.to(device), mask.to(device), bins.to(device)
            logits = model.forward(x, mask)
            probability = model.probability(logits, mask, bins, logarithmic=True)
            probs.append(probability.cpu().numpy())
            n_const.append(mask.sum(dim=-1).cpu().numpy() - 1)

    results = {
        "probs": np.concatenate(probs, 0),
        "n_const": np.concatenate(n_const, 0),
    }

    return results


def main():
    args = get_args()
    set_seeds(args.seed)
    model = torch.load(args.model, map_location="cpu")
    loader = load_data(
        args.data,
        args.num_events,
        start_token=True,
        end_token=True,
        limit_const=False,
        num_const=args.num_const,
        shuffle=False,
    )
    results = get_probs(model=model, loader=loader)
    dir = os.path.dirname(args.model)
    np.savez(os.path.join(dir, f"results_{args.tag}.npz"), **results)


if __name__ == "__main__":
    main()
