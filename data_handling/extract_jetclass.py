# %%
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
import os
import pandas as pd
import vector
import time
from multiprocessing import Pool

start = time.time()

vector.register_awkward()

# %%
# Get all root files corresponding to a given label within a folder
files = []
label = "TTBar_"
folder = "inputFiles/JetClass/part1"
for moth, sub, fs in os.walk(folder):
    for f in fs:
        if f.startswith(label) and f.endswith(".root"):
            files.append(os.path.join(moth, f))

files = sorted(files)
files


# %%
def open_rootFile(filename):
    # Open root file and return the tree as an awkward array
    tree = uproot.open(filename)["tree"]
    table = tree.arrays()
    return table


def get_tuples(table):
    # From the awkward array calculate pt, eta and phy

    # pT and p for all particles
    part_pt = (table["part_px"] ** 2 + table["part_py"] ** 2) ** (1.0 / 2)
    part_p = (
        table["part_px"] ** 2 + table["part_py"] ** 2 + table["part_pz"] ** 2
    ) ** (1.0 / 2)

    # delta eta between particle and the jet
    part_eta_dir = (
        0.5 * np.log((part_p + table["part_pz"]) / (part_p - table["part_pz"]))
        - table["jet_eta"]
    )
    # delta phi between particle and the jet
    part_phi_dir = (
        np.arctan2(table["part_py"], table["part_px"]) - table["jet_phi"] + np.pi
    ) % (2 * np.pi) - np.pi

    # Relative transverse momentum with respect to the jet pT
    part_pt = part_pt / table["jet_pt"]
    return part_pt, part_eta_dir, part_phi_dir


def to_numpy_array(tuple, n_max=200):
    # Transform awkward arrays to zero padded numpy array, tuple contains (pT, eta, phi)
    # where each is an awkward array
    constituents = np.zeros((len(tuple[0]), n_max, 3), dtype=np.float32)
    i = 0
    for pt, eta, phi in zip(*tuple):
        n_const = len(pt)
        if n_const > n_max:
            constituents[i, :, 0] = pt[:200]
            constituents[i, :, 1] = eta[:200]
            constituents[i, :, 2] = phi[:200]
        else:
            constituents[i, :n_const, 0] = pt
            constituents[i, :n_const, 1] = eta
            constituents[i, :n_const, 2] = phi
        i += 1
    return constituents


def to_hdf_file(constituents, out_file):
    # Transform numpy array into pandas dataframe and write to hdf file
    jets, consts, features = constituents.shape
    cols = [
        item
        for sublist in [f"PT_{i},Eta_{i},Phi_{i}".split(",") for i in range(200)]
        for item in sublist
    ]
    df = pd.DataFrame(constituents.reshape((jets, consts * features)), columns=cols)

    df.to_hdf(out_file, key="raw", mode="a", complevel=9)


# %%

out_file = f"{label}test.h5"
i = 1
while os.path.isfile(out_file):
    tmp = out_file.split(".")[0]
    if tmp[-1].isnumeric():
        tmp = "_".join(tmp.split("_")[:-1])
    out_file = tmp + f"_{i}.h5"
    i += 1


def do_it_all(x):
    tab = open_rootFile(x)
    tup = get_tuples(tab)
    return to_numpy_array(tup, 200)


with Pool(5) as p:
    constituents = p.map(do_it_all, files)

constituents = np.concatenate(constituents, axis=0)
print(f"Getting constituents took {int(time.time() - start)} s")

# %%
print(constituents.shape)
to_hdf_file(constituents, out_file=out_file)
print(f"Total time {int(time.time() - start)} s")
