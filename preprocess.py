import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
import ROOT


def preprocess_dataframe(
    df,
    num_features,
    num_bins,
    num_const,
    to_tensor=True,
    reverse=False,
    start=False,
    end=False,
    limit_nconst=False,
):
    x = df.to_numpy(dtype=np.int64)[:, : num_const * num_features]
    x = x.reshape(x.shape[0], -1, num_features)
    padding_mask = x[:, :, 0] != -1

    if limit_nconst:
        keepings = padding_mask.sum(-1) >= num_const
        x = x[keepings]
        padding_mask = padding_mask[keepings]

    if reverse:
        print("Reversing pt order")
        x[x == -1] = np.max(num_bins) + 10
        idx_sort = np.argsort(x[:, :, 0], axis=-1)
        for i in range(len(x)):
            x[i] = x[i, idx_sort[i]]
        x[x == np.max(num_bins) + 10] = -1

    num_prior_bins = np.cumprod((1,) + num_bins[:-1])
    bins = (x * num_prior_bins.reshape(1, 1, num_features)).sum(axis=2)

    if start:
        print("Adding start particles")
        bins = np.concatenate(
            (np.ones((len(bins), 1), dtype=int) * -100, bins),
            axis=1,
        )

        x = np.concatenate(
            (
                np.zeros((len(x), 1, num_features), dtype=int),
                x,
            ),
            axis=1,
        )
        padding_mask = x[:, :, 0] != -1
        bins[~padding_mask] = -100
    else:
        bins[~padding_mask] = -100

    if end:
        print("Adding stop token")
        seq_lengths = padding_mask.sum(-1)
        x = np.append(x, -np.ones((x.shape[0], 1, x.shape[2]), dtype=int), axis=1)
        x[np.arange(x.shape[0]), seq_lengths] = 0
        x = x[:, :-1]
        bins = np.append(bins, -100 * np.ones((bins.shape[0], 1)).astype(int), axis=1)
        bins[np.arange(bins.shape[0]), seq_lengths] = np.prod(num_bins)
        bins = bins[:, :-1]
        padding_mask = x[:, :, 0] != -1

    if to_tensor:
        x = torch.tensor(x)
        padding_mask = torch.tensor(padding_mask)
        bins = torch.tensor(bins)
    print(f"Shapes: {x.shape=} {padding_mask.shape=} {bins.shape=}")
    return x, padding_mask, bins


def imagePreprocessing(jets, filename=None):
    def center():
        mean_eta = np.average(constituents[:, 1], weights=constituents[:, 0])
        mean_phi = np.average(constituents[:, 2], weights=constituents[:, 0])

        constituents[:, 2] -= mean_phi
        constituents[:, 1] -= mean_eta

    def rotate():
        # Calculate the major axis
        eta_coords = (constituents[:, 1] - 15) * constituents[:, 0]
        phi_coords = (constituents[:, 2] - 15) * constituents[:, 0]
        coords = np.vstack([eta_coords, phi_coords])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        sorted_indices = np.argsort(evals)[::-1]
        major_axis = evecs[:, sorted_indices[0]]

        # Rotate major axis to have 0 phi
        theta = np.arctan(major_axis[0] / major_axis[1])
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, s], [-s, c]])
        constituents[:, 1:3] = np.matmul(constituents[:, 1:3], rotation)

    def flip():
        quad1 = 0
        quad2 = 0
        quad3 = 0
        quad4 = 0

        for i in range(len(constituents)):
            if constituents[i, 1] > 0:
                if constituents[i, 2] > 0:
                    quad1 += constituents[i, 0]
                else:
                    quad2 += constituents[i, 0]
            else:
                if constituents[i, 2] > 0:
                    quad3 += constituents[i, 0]
                else:
                    quad4 += constituents[i, 0]

        quad = np.argmax([quad1, quad2, quad3, quad4])

        if quad == 1:
            constituents[:, 2] *= -1
        elif quad == 2:
            constituents[:, 1] *= -1
        elif quad == 3:
            constituents[:, 1] *= -1
            constituents[:, 2] *= -1

    print("Started advancedPreProcess")
    # Loop over all jets
    for i in tqdm(range(np.shape(jets)[0])):
        constituents = jets[i]

        center()
        rotate()
        flip()

        # Normalise pT of the jet to 1
        constituents[:, 0] /= np.sum(constituents[:, 0])
        jets[i] = constituents

    print(f"Exiting advancedPreProcess, shape: {np.shape(jets)}")

    return jets


def discretize_data(
    class_label: int,
    tag: str,
    input_file: str,
    output_file: str,
    lower_q: str,
    upper_q: str,
    nBins: list[int],
    nJets=None,
):
    def read_input():
        es = [f"E_{i}" for i in range(200)]
        px = [f"PX_{i}" for i in range(200)]
        py = [f"PY_{i}" for i in range(200)]
        pz = [f"PZ_{i}" for i in range(200)]
        cols = [item for sublist in zip(es, px, py, pz) for item in sublist]
        print(input_file) 
        #print(pd.read_hdf(input_file))

        df = pd.read_hdf(
            input_file,
            key="table",
            stop=nJets,
        )
        df = df[df["is_signal_new"] == class_label]
        df = df[cols]
        data = df.to_numpy()
        data = data.reshape((-1, 200, 4))

        return data

    def calculate_features(momenta):
        jets = data.sum(1)
        jets_p = np.sqrt(np.square(jets[:, 1:]).sum(1))
        # jets_pt = np.sqrt(np.square(jets[:, 1:3]).sum(1))
        jets_phi = np.arctan2(jets[:, 2], jets[:, 1])
        jets_eta = 0.5 * np.log((jets_p + jets[:, 3]) / (jets_p - jets[:, 3]))

        const_p = np.sqrt(np.square(momenta[:, :, 1:]).sum(2))
        const_pt = np.sqrt(np.square(momenta[:, :, 1:3]).sum(2))
        const_phi = np.arctan2(momenta[:, :, 2], momenta[:, :, 1])
        const_eta = 0.5 * np.log(
            (const_p + momenta[:, :, 3]) / (const_p - momenta[:, :, 3])
        )

        d_eta = const_eta - jets_eta[..., np.newaxis]
        d_phi = const_phi - jets_phi[..., np.newaxis]
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi

        d_eta[const_pt == 0] = 0
        d_phi[const_pt == 0] = 0

        return const_pt, d_eta, d_phi

    def check_pt_oredering(pts):
        for i in range(len(pts)):
            assert np.all(pts[i, :-1] >= pts[i, 1:]), "Data not sorted in pT"

    def get_binning():
        # If QCD training as input, get the bins
        if input_file.split("/")[-1] == "train.h5" and class_label == 0:
            pt_bins = np.linspace(
                np.quantile(np.log(const_pt[const_pt != 0]), lower_q),
                np.quantile(np.log(const_pt[const_pt != 0]), upper_q),
                nBins[0],
            )
            eta_bins = np.linspace(-0.8, 0.8, nBins[1])
            phi_bins = np.linspace(-0.8, 0.8, nBins[2])

            if not os.path.isdir("preprocessing_bins"):
                os.makedirs("preprocessing_bins")

            np.save(f"preprocessing_bins/pt_bins_{tag}", pt_bins)
            np.save(f"preprocessing_bins/eta_bins_{tag}", eta_bins)
            np.save(f"preprocessing_bins/phi_bins_{tag}", phi_bins)
            print("Created bins\n")
        # Else load the binning according to given tag
        else:
            pt_bins = np.load(f"preprocessing_bins/pt_bins_{tag}.npy")
            eta_bins = np.load(f"preprocessing_bins/eta_bins_{tag}.npy")
            phi_bins = np.load(f"preprocessing_bins/phi_bins_{tag}.npy")
            print(f"\nLoaded bins with tag {tag}\n")
        return pt_bins, eta_bins, phi_bins

    def discretize():
        # Get the discrete values
        const_pt_disc = np.digitize(np.log(const_pt), pt_bins).astype(np.int16)
        d_eta_disc = np.digitize(d_eta, eta_bins).astype(np.int16)
        d_phi_disc = np.digitize(d_phi, phi_bins).astype(np.int16)

        # Apply mask
        const_pt_disc[const_pt == 0] = -1
        d_eta_disc[const_pt == 0] = -1
        d_phi_disc[const_pt == 0] = -1
        return const_pt_disc, d_eta_disc, d_phi_disc

    def get_df(pt, eta, phi):
        stacked = np.stack([pt, eta, phi], -1)
        stacked = stacked.reshape((-1, 600))
        cols = [
            item
            for sublist in [f"PT_{i},Eta_{i},Phi_{i}".split(",") for i in range(200)]
            for item in sublist
        ]
        df = pd.DataFrame(stacked, columns=cols)
        return df

    print(f"Input: {input_file}\nOutput: {output_file}")

    data = read_input()
    print(f"Data shape: {data.shape}\n")
    const_pt, d_eta, d_phi = calculate_features(data)
    check_pt_oredering(const_pt)


    labelType = "qcd"
    if args.class_label == 1:
      labelType = "top"
    file = ROOT.TFile.Open("originalJets_%s.root"%labelType, "RECREATE")
    tree = ROOT.TTree("tree","tree")

    constit_pt = ROOT.std.vector[float]()
    tree.Branch("constit_pt", constit_pt)
    constit_eta = ROOT.std.vector[float]()
    tree.Branch("constit_eta", constit_eta)
    constit_phi = ROOT.std.vector[float]()
    tree.Branch("constit_phi", constit_phi)

    for cjet_pt, cjet_eta, cjet_phi in zip(const_pt, d_eta, d_phi):
        # Clear the contents of the vector
        constit_pt.clear()
        constit_eta.clear()
        constit_phi.clear()
        # Replace the contents in the vector with the contents
        # from the current array
        mask = cjet_pt > 0
        nconstit = len(cjet_pt[mask])
        constit_pt.reserve(len(cjet_pt))
        constit_eta.reserve(len(cjet_pt))
        constit_phi.reserve(len(cjet_pt))
        for cpt, ceta, cphi in zip(cjet_pt, cjet_eta, cjet_phi):
            if(cpt > 0):
              constit_pt.push_back(cpt)
              constit_eta.push_back(ceta)
              constit_phi.push_back(cphi)

        tree.Fill()

    file.WriteObject(tree, "tree")
    file.Close()



    pt_bins, eta_bins, phi_bins = get_binning()
    const_pt_disc, d_eta_disc, d_phi_disc = discretize()
 
    print("testing shape: ", const_pt.shape)

    print(f"\npT bin range: {const_pt_disc[const_pt!=0].min()} {const_pt_disc.max()}")
    print(f"eta bin range: {d_eta_disc[const_pt!=0].min()} {d_eta_disc.max()}")
    print(f"phi bin range: {d_phi_disc[const_pt!=0].min()} {d_phi_disc.max()}\n")

    # Collect continuous data in dataframe
    raw = get_df(const_pt, d_eta, d_phi)
    disc = get_df(const_pt_disc, d_eta_disc, d_phi_disc)

    # Write dataframes into compressed hdf5 file
    raw.to_hdf(output_file, key="raw", mode="w", complevel=9)
    disc.to_hdf(output_file, key="discretized", mode="r+", complevel=9)

    print("\nDiscretized dataframe description")
    print(disc.describe())


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--class_label", type=int, choices=[0, 1])
    parser.add_argument("--tag", type=str)
    parser.add_argument("--nBins", "-n", type=int, nargs=3)
    parser.add_argument("--input_file", "-I", type=str)
    parser.add_argument("--lower_q", "-l", type=float, default=0.0)
    parser.add_argument("--upper_q", "-u", type=float, default=1.0)
    parser.add_argument("--nJets", "-N", type=int, default=None)
    args = parser.parse_args()

    train_test = args.input_file.split("/")[-1][:-3]
    print(f"Dataset: {train_test}")
    output_path = os.path.join(os.path.dirname(args.input_file), "discretized")
    if not os.path.exists(output_path):
        print("\nCreating output path\n")
        os.makedirs(output_path)
    output_file = f"{train_test}_{['qcd', 'top'][args.class_label]}_{args.tag}.h5"
    output_file = os.path.join(output_path, output_file)

    discretize_data(
        class_label=args.class_label,
        tag=args.tag,
        input_file=args.input_file,
        output_file=output_file,
        lower_q=args.lower_q,
        upper_q=args.upper_q,
        nBins=args.nBins,
        nJets=args.nJets,
    )
