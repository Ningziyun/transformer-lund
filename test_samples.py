# %%
from model import CNNclass
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from preprocess import imagePreprocessing
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--bg", "-b", type=str)
parser.add_argument("--sig", "-s", type=str)
parser.add_argument("--save_dir", type=str, default="samples_test/test")
parser.add_argument("--num_jets", "-N", type=int, default=100000)
parser.add_argument("--num_const", "-c", type=int, default=50)
parser.add_argument("--num_epochs", "-E", type=int, default=30)
parser.add_argument("--continuous", action="store_true")
args = parser.parse_args()

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

torch.manual_seed(0)
np.random.seed(0)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# %%
def make_continues(jets):
    pt_bins = np.load("preprocessing_bins/pt_bins_pt80_eta60_phi60_lower001.npy")
    eta_bins = np.load("preprocessing_bins/eta_bins_pt80_eta60_phi60_lower001.npy")
    phi_bins = np.load("preprocessing_bins/phi_bins_pt80_eta60_phi60_lower001.npy")

    pt_disc = jets[:, :, 0]
    mask = pt_disc == 0
    eta_disc = jets[:, :, 1]
    phi_disc = jets[:, :, 2]

    pt_con = (pt_disc - np.random.uniform(0.0, 1.0, size=pt_disc.shape)) * (
        pt_bins[1] - pt_bins[0]
    ) + pt_bins[0]
    eta_con = (eta_disc - np.random.uniform(0.0, 1.0, size=eta_disc.shape)) * (
        eta_bins[1] - eta_bins[0]
    ) + eta_bins[0]
    phi_con = (phi_disc - np.random.uniform(0.0, 1.0, size=phi_disc.shape)) * (
        phi_bins[1] - phi_bins[0]
    ) + phi_bins[0]

    continues_jets = np.stack((np.exp(pt_con), eta_con, phi_con), -1)
    continues_jets[mask] = 0

    return continues_jets


def load_data(file, continuous=False, orig=False):
    if file.endswith("npz"):
        dat = np.load(file)["jets"][: args.num_jets, : args.num_const]
    elif file.endswith("h5"):
        if not orig:
            dat = pd.read_hdf(file, key="discretized", stop=args.num_jets)
            dat = dat.to_numpy(dtype=np.int64)[:, : args.num_const * 3]
            dat = dat.reshape(dat.shape[0], -1, 3)
        else:
            print("Loading raw")
            dat = pd.read_hdf(file, key="raw", stop=args.num_jets)
            dat = dat.to_numpy()[:, : args.num_const * 3]
            dat = dat.reshape(dat.shape[0], -1, 3)
    else:
        assert False, "Filetype for not supported"
    dat = np.delete(dat, np.where(dat[:, 0, 0] == 0)[0], axis=0)
    dat[dat == -1] = 0

    if continuous:
        dat = make_continues(dat)
    return dat


def plot_data_dist():
    fig, axes = plt.subplots(1, 3, constrained_layout=True, figsize=(16, 9))
    nBins = 40
    for i in range(3):
        if i > 0:
            nBins = 30
        axes[i].hist(
            bg[:, :, i][bg[:, :, 0] != 0].flatten(),
            histtype="step",
            bins=np.linspace(-0.5, nBins + 0.5, nBins + 2),
            density=True,
            label="Background",
        )
        axes[i].hist(
            sig[:, :, i][sig[:, :, 0] != 0].flatten(),
            histtype="step",
            bins=np.linspace(-0.5, nBins + 0.5, nBins + 2),
            density=True,
            label="Signal",
        )
        if i == 0:
            axes[i].set_yscale("log")
            axes[i].legend()
    fig.savefig(os.path.join(args.save_dir, "distributions.png"))


def jets_to_images(data):
    jets = imagePreprocessing(data.astype(float))
    images = np.zeros((len(jets), 30, 30))
    bins = np.linspace(-0.8, 0.8, 31) if args.continuous else np.arange(-15.5, 15.5, 1)
    for i in tqdm(range(len(jets))):
        images[i], _, _ = np.histogram2d(
            jets[i, :, 1],
            jets[i, :, 2],
            bins=(bins, bins),
            weights=jets[i, :, 0],
        )
    images = images[:, np.newaxis, ...]

    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    for i in range(2):
        ax[i].imshow(images[labels == i].mean(0).squeeze(), cmap="Blues")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(["Bg", "Sig"][i])
    fig.savefig(os.path.join(args.save_dir, "mean_images.png"))

    return images


def get_dataloader():
    train_data = torch.tensor(data[: int(0.75 * len(data))], dtype=torch.float32)
    train_labels = torch.tensor(labels[: int(0.75 * len(data))], dtype=torch.float32)

    val_data = torch.tensor(data[int(0.75 * len(data)) :], dtype=torch.float32)
    val_labels = torch.tensor(labels[int(0.75 * len(data)) :], dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(train_data, train_labels[..., np.newaxis])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels[..., np.newaxis])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
    )
    return dataloader, val_dataloader


def train_model():
    losses = []
    val_losses = []
    min_loss = np.inf
    global_step = 0
    for epoch in tqdm(range(args.num_epochs)):
        classi.train()
        for x, y in dataloader:
            global_step += 1
            opt.zero_grad()

            x = x.to(device)
            y = y.to(device)

            pred = classi(x)
            loss = classi.loss(pred, y)
            losses.append([global_step, loss.cpu().detach().numpy()])

            loss.backward()
            opt.step()

        with torch.no_grad():
            classi.eval()
            tmp = []
            for x, y in val_dataloader:
                x = x.to(device)
                y = y.to(device)
                loss = classi.loss(classi(x), y)
                tmp.append(loss.cpu().detach().numpy())
            val_losses.append([global_step, np.mean(tmp)])
            if np.mean(tmp) < min_loss:
                min_loss = np.mean(tmp)
                torch.save(classi, os.path.join(args.save_dir, "classifier_best.pt"))

    losses = np.array(losses)
    val_losses = np.array(val_losses)
    np.savez(
        os.path.join(args.save_dir, "training"), losses=losses, val_losses=val_losses
    )
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(losses[:, 0], losses[:, 1], alpha=0.2, c="b")
    ax.plot(
        losses[99:, 0],
        moving_average(losses[:, 1], 100),
        alpha=1.0,
        label="Train",
        c="b",
    )
    ax.plot(val_losses[:, 0], val_losses[:, 1], label="Val")
    ax.legend()
    fig.savefig(os.path.join(args.save_dir, "training.png"))


bg = load_data(args.bg, args.continuous)
print(f"Got bg from {args.bg}")
sig = load_data(args.sig, args.continuous)
print(f"Got sig from {args.sig}")


print(bg.shape, sig.shape)
# %%
data = np.append(bg, sig, axis=0)
labels = np.append(np.zeros(len(bg)), np.ones(len(sig)))
plot_data_dist()

# %%
data = jets_to_images(data)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

classi = CNNclass()
idcs = np.random.permutation(len(data))
data = data[idcs]
labels = labels[idcs]

dataloader, val_dataloader = get_dataloader()

# %%
classi.to(device)
opt = torch.optim.Adam(
    classi.parameters(),
    lr=0.0005,
)

train_model()

# %%
labels = []
preds = []
classi.eval()
for x, y in val_dataloader:
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        pred = classi(x)
        preds.append(pred.cpu().detach().numpy())
        labels.append(y.cpu().detach().numpy())

torch.save(classi, os.path.join(args.save_dir, "classifier_last.pt"))
preds = np.concatenate(preds, axis=0)
labels = np.concatenate(labels, axis=0)
np.savez(os.path.join(args.save_dir, "predictions_last"), labels=labels, preds=preds)

# %%
fpr, tpr, _ = roc_curve(y_true=labels, y_score=preds)
auc = roc_auc_score(y_true=labels, y_score=preds)

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(tpr, 1.0 / fpr, label=f"Last AUC: {auc:.3f}")
ax.plot(np.linspace(0, 1, 1000), 1.0 / np.linspace(0, 1, 1000), color="grey")

classi = torch.load(os.path.join(args.save_dir, "classifier_best.pt"), weights_only=False)
labels = []
preds = []
classi.eval()
for x, y in val_dataloader:
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        pred = classi(x)
        preds.append(pred.cpu().detach().numpy())
        labels.append(y.cpu().detach().numpy())

preds = np.concatenate(preds, axis=0)
labels = np.concatenate(labels, axis=0)
np.savez(os.path.join(args.save_dir, "predictions_best"), labels=labels, preds=preds)

fpr, tpr, _ = roc_curve(y_true=labels, y_score=preds)
auc = roc_auc_score(y_true=labels, y_score=preds)

ax.plot(tpr, 1.0 / fpr, label=f"Best AUC: {auc:.3f}")
ax.plot(np.linspace(0, 1, 1000), 1.0 / np.linspace(0, 1, 1000), color="grey")

ax.set_yscale("log")
ax.grid(which="both")
ax.set_ylim(0.9, 1e5)
ax.legend()
fig.savefig(os.path.join(args.save_dir, "roc.png"))

# %%
fig, ax = plt.subplots(constrained_layout=True)
ax.hist([preds[labels == 0], preds[labels == 1]], bins=100, histtype="step")
ax.set_yscale("log")
fig.savefig(os.path.join(args.save_dir, "predictions.png"))
