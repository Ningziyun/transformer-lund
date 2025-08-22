import numpy as np
import sys, json, os
import pandas as pd


def transform_momenta(momenta, mask):
    pts = momenta[:, :, 0]
    etas = momenta[:, :, 1]
    phis = momenta[:, :, 2]

    drs = np.sqrt(np.sum(np.square(momenta[:, :, 1:3]), -1))

    pxs = np.cos(phis) * pts
    pys = np.sin(phis) * pts
    pxs[~mask] = 0
    pys[~mask] = 0

    pxj = np.sum(pxs, -1)
    pyj = np.sum(pys, 1)
    ptj = np.sqrt(pxj**2 + pyj**2)

    etas[~mask] = 0
    phis[~mask] = 0
    logpts = np.log(pts)
    logpts[~mask] = 0
    logpt_ptj = np.log(pts / ptj.reshape(-1, 1))
    logpt_ptj[~mask] = 0
    drs[~mask] = 0

    newVec = np.stack(
        [etas, phis, logpts, logpt_ptj, drs],
        -1,
    )
    return newVec


def get_config(test=False):
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        config = json.load(open(sys.argv[1], "r"))
    else:
        assert not test, f"Provide config file used for training"
        config = json.load(open("config.json", "r"))

    if not test:
        i = 0
        original = config["logging"]["logfolder"]
        while os.path.isdir(config["logging"]["logfolder"]):
            i += 1
            config["logging"]["logfolder"] = f"{original}_{i}"

        os.makedirs(config["logging"]["logfolder"])
        json.dump(
            config,
            open(os.path.join(config["logging"]["logfolder"], "config.json"), "w"),
            sort_keys=True,
            indent=2,
        )

    return config


def make_continues(jets, mask, noise=False):
    pt_bins = np.load("../preprocessing_bins/pt_bins_pt40_eta30_phi30_lower001.npy")
    eta_bins = np.load("../preprocessing_bins/eta_bins_pt40_eta30_phi30_lower001.npy")
    phi_bins = np.load("../preprocessing_bins/phi_bins_pt40_eta30_phi30_lower001.npy")

    pt_disc = jets[:, :, 0]
    eta_disc = jets[:, :, 1]
    phi_disc = jets[:, :, 2]

    if noise:
        pt_con = (pt_disc - np.random.uniform(0.0, 1.0, size=pt_disc.shape)) * (
            pt_bins[1] - pt_bins[0]
        ) + pt_bins[0]
        eta_con = (eta_disc - np.random.uniform(0.0, 1.0, size=eta_disc.shape)) * (
            eta_bins[1] - eta_bins[0]
        ) + eta_bins[0]
        phi_con = (phi_disc - np.random.uniform(0.0, 1.0, size=phi_disc.shape)) * (
            phi_bins[1] - phi_bins[0]
        ) + phi_bins[0]
    else:
        pt_con = (pt_disc - 0.5) * (pt_bins[1] - pt_bins[0]) + pt_bins[0]
        eta_con = (eta_disc - 0.5) * (eta_bins[1] - eta_bins[0]) + eta_bins[0]
        phi_con = (phi_disc - 0.5) * (phi_bins[1] - phi_bins[0]) + phi_bins[0]


    pt_con = np.exp(pt_con)
    pt_con[~mask] = 0.0
    eta_con[~mask] = 0.0
    phi_con[~mask] = 0.0

    return np.stack((pt_con, eta_con, phi_con), -1)


def load_data(params, test=False, plot_dists=None,):
    def load_file(file, key=None):
        # Load  sample file
        if file.endswith("npz"):
            if test:
                file = file.replace("train_", "test_")
            print(f"\nChanged file to {file}")
            dat = np.load(file)["jets"][: params["n_jets"], : params["n_const"]]
            mask = dat.sum(-1) != 0
        # Load data file
        elif file.endswith("h5"):
            if test:
                file = file.replace("val", "test")
                file = file.replace("train", "test")
                print(f"\nChanged file to {file}")
                dat = pd.read_hdf(
                    file,
                    key=key,
                    stop=params["n_jets"],
                )
            else:
                dat = pd.read_hdf(file, key=key, stop=params["n_jets"])
            dat = dat.to_numpy()[:, : params["n_const"] * 3]
            dat = dat.reshape(dat.shape[0], -1, 3)
            mask = dat[:, :, 0] != -1
        else:
            assert False, "Filetype for bg not supported"
        # Delete empty jest (can occur in samplings)
        mask = np.delete(mask, np.where(dat[:, 0, 0] == 0)[0], axis=0)
        dat = np.delete(dat, np.where(dat[:, 0, 0] == 0)[0], axis=0)
        dat[dat == -1] = 0
        return dat, mask

    print(params["bg_file"])
    print(params["sig_file"])

    bg, bg_mask = load_file(params["bg_file"], key=params["bg_key"])
    sig, sig_mask = load_file(params["sig_file"], key=params["sig_key"])

    if params["bg_key"] == "discretized":
        print(f"BG made continuous, with noise {params['bg_noise']}\n")
        bg = make_continues(bg, mask=bg_mask, noise=params["bg_noise"],)
    else:
        bg[~bg_mask] = 0

    if params["sig_key"] == "discretized":
        print(f"Sig made continuous, with noise {params['bg_noise']}\n")
        sig = make_continues(sig, mask=sig_mask, noise=params["sig_noise"])
    else:
        sig[~sig_mask] = 0

    print(f"Bg shape {bg.shape}, Sig shape {sig.shape}\n")

    bg = transform_momenta(bg, mask=bg_mask)
    sig = transform_momenta(sig, mask=sig_mask)
    data = np.append(bg, sig, 0)
    labels = np.append(np.zeros(len(bg)), np.ones(len(sig)))
    shuffle = np.random.permutation(len(data))

    if not plot_dists is None:
        import matplotlib.pyplot as plt

        features = data.shape[-1]
        fig, axes = plt.subplots(
            features, 1, constrained_layout=True, figsize=(features * 3, 10)
        )
        for i in range(data.shape[-1]):
            range_min = data[:, :, i].min()
            range_max = data[:, :, i].max()
            axes[i].hist(
                bg[bg_mask, i],
                bins=300,
                range=[range_min, range_max],
                histtype="step",
                density=True,
                label="Background",
            )
            axes[i].hist(
                sig[sig_mask, i],
                bins=300,
                range=[range_min, range_max],
                histtype="step",
                density=True,
                label="Signal",
            )
        axes[0].legend()
        fig.savefig(plot_dists)
        plt.close(fig)

    return data[shuffle], labels[shuffle]


if __name__ == "__main__":
    pass
