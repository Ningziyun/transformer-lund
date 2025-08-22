import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from preprocess import imagePreprocessing, preprocess_dataframe
import pandas as pd


def get_samples(
    model,
    data_loader,
    device,
    trunc=None,
):
    if type(model) is str:
        model = torch.load(model)
    model.to(device)
    model.classifier = False

    samples = []
    bins = []
    for x, _, _ in tqdm(iterable=data_loader, desc="Batch", total=len(data_loader)):
        _samples, _sample_bins = model.sample(
            x[:, 0].to(device),
            device,
            len_seq=x.size(1),
            trunc=trunc,
        )
        samples.append(_samples.cpu())
        bins.append(_sample_bins.cpu())
    samples = torch.concat(samples, 0).cpu().numpy()
    bins = torch.concat(bins, 0).cpu().numpy()
    model.cpu()
    return samples, bins


def jets_to_images(data):
    jets = imagePreprocessing(data.astype(float))
    print(jets.shape)
    images = np.zeros((len(jets), 30, 30)).astype(np.float32)
    bins = (np.arange(-15.5, 15.5, 1), np.arange(-15.5, 15.5, 1))
    for i in tqdm(range(len(jets))):
        tmp, jets = jets[0], jets[1:]
        images[i], _, _ = np.histogram2d(
            tmp[:, 1],
            tmp[:, 2],
            bins=bins,
            weights=tmp[:, 0],
        )
    return images


def get_data(
    N: int,
    files: list[str],
    tags: list[str],
    reverse=False,
    start=False,
    end=False,
    newF=False,
    num_const=20,
    limit_nconst=True,
):
    key = "discretized2" if newF else "discretized"
    assert len(files) == len(
        tags
    ), f"Need same number of tags and files (given {len(tags)} {len(files)}"
    data_loaders = {}
    orig_data = {}
    for ind, file in enumerate(files):
        df = pd.read_hdf(
            file,
            key=key,
            stop=N,
        )
        jets, mask, bins = preprocess_dataframe(
            df,
            num_features=3,
            num_bins=(41, 31, 31),
            num_const=num_const,
            limit_nconst=limit_nconst,
            reverse=reverse,
            start=start,
            end=end,
        )
        loader = DataLoader(
            TensorDataset(jets, mask, bins), batch_size=100, shuffle=False
        )
        data_loaders[tags[ind]] = loader
        orig_data[tags[ind]] = (jets, mask, bins)

    return data_loaders, orig_data


def idx_to_bins(x):
    pT = x % 41
    eta = (x - pT) // 41 % (1271 // 41)
    phi = (x - pT - 41 * eta) // 1271
    return pT, eta, phi


if __name__ == "__main__":
    pass
