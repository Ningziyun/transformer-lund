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
    sample_name: str,
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
        """
        Compute (pT, d_eta, d_phi) for each constituent.

        This version is numerically safe:
        - Avoid division-by-zero / log of non-positive values in eta
        - Replace any NaN/inf with 0 so downstream steps stay stable
        - Treat invalid constituents as padding (pt=0, angles=0)
        """
        def safe_eta(p, pz, eps=1e-12):
            denom = p - pz
            numer = p + pz
            out = np.zeros_like(p, dtype=np.float64)

            valid = (np.abs(denom) > eps) & (numer > 0) & (denom > 0) \
                    & np.isfinite(p) & np.isfinite(pz)
            out[valid] = 0.5 * np.log(numer[valid] / denom[valid])
            return out

        jets = data.sum(1)
        jets_p   = np.sqrt(np.square(jets[:, 1:]).sum(1))
        jets_phi = np.arctan2(jets[:, 2], jets[:, 1])
        jets_eta = safe_eta(jets_p, jets[:, 3])

        const_p   = np.sqrt(np.square(momenta[:, :, 1:]).sum(2))
        const_pt  = np.sqrt(np.square(momenta[:, :, 1:3]).sum(2))
        const_phi = np.arctan2(momenta[:, :, 2], momenta[:, :, 1])
        const_eta = safe_eta(const_p, momenta[:, :, 3])

        d_eta = const_eta - jets_eta[..., np.newaxis]
        d_phi = const_phi - jets_phi[..., np.newaxis]
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi

        # Mark invalid constituents as padding
        valid_const = (const_pt > 0) & np.isfinite(const_pt) & np.isfinite(d_eta) & np.isfinite(d_phi)
        const_pt = np.where(valid_const, const_pt, 0.0)
        d_eta    = np.where(valid_const, d_eta, 0.0)
        d_phi    = np.where(valid_const, d_phi, 0.0)

        # Final safety: remove any remaining NaN/inf
        const_pt = np.nan_to_num(const_pt, nan=0.0, posinf=0.0, neginf=0.0)
        d_eta    = np.nan_to_num(d_eta,    nan=0.0, posinf=0.0, neginf=0.0)
        d_phi    = np.nan_to_num(d_phi,    nan=0.0, posinf=0.0, neginf=0.0)

        return const_pt, d_eta, d_phi
    '''
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
    '''

    def check_pt_oredering(pts):
        for i in range(len(pts)):
            assert np.all(pts[i, :-1] >= pts[i, 1:]), "Data not sorted in pT"

    def dedup_within_event(const_pt, d_eta, d_phi,
                        deta_tol=1e-3, dphi_tol=1e-3):
        """
        Deduplicate near-identical constituents within each event.

        Key idea:
        - The Lund-plane pathology (very large log(1/dR) and very negative log(kt))
        often comes from merges with dR ~ 0.
        - Two constituents can have (almost) identical direction but different pt.
        If the dedup key includes log(pt), those will NOT be removed and can
        still generate dR~0 merges downstream.
        - Here we deduplicate by (d_eta, d_phi) only, and MERGE pt into the first
        occurrence instead of inserting zeros in the middle or dropping energy.

        Behavior:
        - Preserve original ordering as much as possible.
        - Keep the first occurrence; if another constituent falls into the same
        (d_eta, d_phi) bin, add its pt to the kept one.
        - Pack kept constituents to the front; tail is padding (pt=0, eta=0, phi=0).
        """
        pt_in  = const_pt
        eta_in = d_eta
        phi_in = d_phi

        n_ev, n_const = pt_in.shape

        pt_out  = np.zeros_like(pt_in,  dtype=pt_in.dtype)
        eta_out = np.zeros_like(eta_in, dtype=eta_in.dtype)
        phi_out = np.zeros_like(phi_in, dtype=phi_in.dtype)

        for i in range(n_ev):
            # Map from (quantized eta, quantized phi) -> output index
            seen_idx = {}
            write_idx = 0

            for j in range(n_const):
                p = pt_in[i, j]
                if p <= 0:
                    continue  # padding / invalid

                k1 = int(np.round(eta_in[i, j] / deta_tol))
                k2 = int(np.round(phi_in[i, j] / dphi_tol))
                key = (k1, k2)

                if key in seen_idx:
                    # Same direction bin: merge pt into the first occurrence
                    k = seen_idx[key]
                    pt_out[i, k] = pt_out[i, k] + p
                    continue

                # First time seeing this direction bin: keep it
                seen_idx[key] = write_idx
                pt_out[i, write_idx]  = p
                eta_out[i, write_idx] = eta_in[i, j]
                phi_out[i, write_idx] = phi_in[i, j]
                write_idx += 1

                if write_idx >= n_const:
                    break

        return pt_out, eta_out, phi_out

    def sort_by_pt_desc(const_pt, d_eta, d_phi):
        """
        Sort constituents within each event by descending pT.

        Notes:
        - We treat pt<=0 as padding/invalid and push them to the end.
        - Sorting must be applied consistently to (pt, d_eta, d_phi) to keep alignment.
        """
        pt = const_pt.copy()
        eta = d_eta.copy()
        phi = d_phi.copy()

        # Replace non-positive pT with -inf so they go to the end after sorting
        pt_for_sort = np.where(pt > 0, pt, -np.inf)

        # argsort ascending then reverse => descending
        idx = np.argsort(pt_for_sort, axis=1)[:, ::-1]

        # Fancy indexing to reorder each row
        row = np.arange(pt.shape[0])[:, None]
        pt  = pt[row, idx]
        eta = eta[row, idx]
        phi = phi[row, idx]

        return pt, eta, phi
    
    '''
    def dedup_within_event(const_pt, d_eta, d_phi,
                        logpt_tol=1e-3, deta_tol=1e-3, dphi_tol=1e-3):
        """
        Deduplicate near-identical constituents within each event.

        IMPORTANT:
        - Do NOT set pt=0 "in place" for duplicates, because that inserts zeros in the
        middle of a pt-sorted sequence and breaks monotonic ordering.
        - Instead, keep the first occurrence, then compact (pack) all kept constituents
        to the front while preserving their original order.
        - All dropped entries are moved to the tail and set to padding (pt=0, eta=0, phi=0).
        """
        pt_in  = const_pt
        eta_in = d_eta
        phi_in = d_phi

        n_ev, n_const = pt_in.shape

        # Output arrays: initialized as padding everywhere
        pt_out  = np.zeros_like(pt_in,  dtype=pt_in.dtype)
        eta_out = np.zeros_like(eta_in, dtype=eta_in.dtype)
        phi_out = np.zeros_like(phi_in, dtype=phi_in.dtype)

        for i in range(n_ev):
            seen = set()
            write_idx = 0

            for j in range(n_const):
                p = pt_in[i, j]
                if p <= 0:
                    # Treat non-positive pt as padding/invalid; stop early if your data guarantees padding at tail
                    # (We do not "break" here to stay robust in case padding is not strictly at the tail.)
                    continue

                # Build a quantized key in (log(pt), deta, dphi) space
                logp = np.log(p)
                k0 = int(np.round(logp / logpt_tol))
                k1 = int(np.round(eta_in[i, j] / deta_tol))
                k2 = int(np.round(phi_in[i, j] / dphi_tol))
                key = (k0, k1, k2)

                if key in seen:
                    # Duplicate: drop it (it will remain padding in the output tail)
                    continue

                # First occurrence: keep it and pack to the front
                seen.add(key)
                pt_out[i, write_idx]  = p
                eta_out[i, write_idx] = eta_in[i, j]
                phi_out[i, write_idx] = phi_in[i, j]
                write_idx += 1

                if write_idx >= n_const:
                    break

        return pt_out, eta_out, phi_out
    '''
    '''
    def dedup_within_event(const_pt, d_eta, d_phi,
                           logpt_tol=1e-3, deta_tol=1e-3, dphi_tol=1e-3):
        """
        Deduplicate near-identical constituents within each event.
        Similarity is defined in the feature space used to compute Lund variables:
        (log(pt), d_eta, d_phi). Keep the first occurrence, drop later duplicates.

        Dropped constituents are turned into padding by setting pt=0 (and d_eta/d_phi=0).
        This preserves array shapes (N, 200) with minimal downstream changes.
        """
        # Make copies to avoid surprising side effects
        pt = const_pt.copy()
        eta = d_eta.copy()
        phi = d_phi.copy()

        n_ev, n_const = pt.shape
        for i in range(n_ev):
            seen = set()
            for j in range(n_const):
                if pt[i, j] <= 0:
                    continue  # padding / invalid

                # Build an approximate key from the features that drive kt and deltaR
                # Use quantization so "near-identical" maps to the same key
                logpt = np.log(pt[i, j])
                k0 = int(np.round(logpt / logpt_tol))
                k1 = int(np.round(eta[i, j] / deta_tol))
                k2 = int(np.round(phi[i, j] / dphi_tol))
                key = (k0, k1, k2)

                if key in seen:
                    # Drop duplicates within the same event only
                    pt[i, j] = 0.0
                    eta[i, j] = 0.0
                    phi[i, j] = 0.0
                else:
                    seen.add(key)

        return pt, eta, phi
    '''
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

    '''
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
    '''
    def discretize():
        """
        Discretize continuous features into integer bins.

        Important:
        - const_pt contains padding entries with pt=0.
        - np.log(0) -> -inf, which triggers a RuntimeWarning.
        - We avoid the warning by only taking log for pt>0 entries.
        """
        # Safe log(pt): only compute log where pt>0, keep 0 elsewhere
        log_pt = np.zeros_like(const_pt, dtype=np.float64)
        mask = const_pt > 0
        log_pt[mask] = np.log(const_pt[mask])

        # Digitize using the safe log_pt
        const_pt_disc = np.digitize(log_pt, pt_bins).astype(np.int16)
        d_eta_disc = np.digitize(d_eta, eta_bins).astype(np.int16)
        d_phi_disc = np.digitize(d_phi, phi_bins).astype(np.int16)

        # Apply padding mask
        const_pt_disc[~mask] = -1
        d_eta_disc[~mask] = -1
        d_phi_disc[~mask] = -1

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

    # Enforce pT ordering explicitly (do this BEFORE any checks/dedup)
    const_pt, d_eta, d_phi = sort_by_pt_desc(const_pt, d_eta, d_phi)

    check_pt_oredering(const_pt)

    

    if args.dedup_particles:
        print("Deduplicating near-identical constituents within each event (keep the first)")
        # Tolerances can be tuned; start small to only remove truly duplicated records
        const_pt, d_eta, d_phi = dedup_within_event(
            const_pt, d_eta, d_phi,
            deta_tol=args.deta_tol,
            dphi_tol=args.dphi_tol,
        )
        # Optional: re-check ordering is still valid (pt duplicates are set to 0)
        # IMPORTANT: dedup merges pT and can break ordering -> re-sort
        const_pt, d_eta, d_phi = sort_by_pt_desc(const_pt, d_eta, d_phi)
        
        check_pt_oredering(const_pt)

    labelType = "qcd"
    if args.class_label == 1:
      labelType = "top"
    file = ROOT.TFile.Open("originalJets_%s_%s.root"%(labelType,sample_name), "RECREATE")
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
    parser.add_argument("--dedup_particles", action="store_true",
                    help="Deduplicate near-identical constituents within each event (keep the first).")
    parser.add_argument("--deta_tol", type=float, default=1e-1,
                    help="Tolerance in d_eta for deduplication (continuous space).")
    parser.add_argument("--dphi_tol", type=float, default=1e-1,
                    help="Tolerance in d_phi for deduplication (continuous space).")
    args = parser.parse_args()

    sample_name = args.input_file.split("/")[-1][:-3]
    print(f"Dataset: {sample_name}")
    output_path = os.path.join(os.path.dirname(args.input_file), "discretized")
    if not os.path.exists(output_path):
        print("\nCreating output path\n")
        os.makedirs(output_path)
    output_file = f"{sample_name}_{['qcd', 'top'][args.class_label]}_{args.tag}.h5"
    output_file = os.path.join(output_path, output_file)

    discretize_data(
        class_label=args.class_label,
        tag=args.tag,
        input_file=args.input_file,
        output_file=output_file,
        lower_q=args.lower_q,
        upper_q=args.upper_q,
        nBins=args.nBins,
        sample_name=sample_name,
        nJets=args.nJets,
    )