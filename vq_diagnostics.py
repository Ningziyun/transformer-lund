"""
Diagnostics for the constituent VQ-VAE tokenizer.

Works on the pre-processed per-constituent features from the helpers datasets (dataset.X,
dataset.mask, dataset.jet_data) for --input_format 4vec, relvec or ktdr. Two views:

  pre-processed space   what the VQ-VAE sees and is trained on
  physical space        helpers.undo_preprocess (+ helpers.make_absolute_constituents for relvec):
                          4vec, relvec   jet pT, mass, width, pT_D closure + mean jet image
                                         (+ constituent mass-shell check for 4vec)
                          ktdr           Lund-plane density (ln 1/dR, ln kt)

Called at the end of train_vqvae.py, or standalone (settings taken from the checkpoint):

    python vq_diagnostics.py --tokenizer tokenizer.pt --data test.h5 --out diag_test [--token-stats]

Outputs in <out>/ (metrics.json + PNGs):
  codebook health
    codebook_usage.png              rank-frequency of codes per RVQ level; utilisation, perplexity/K
    level_cooccurrence.png          joint usage of level-0 x level-1 codes (+ normalised MI)
    codebook_geometry.png           latents + level-0 codebook, all pairs of the first 4 principal components
    codebook_geometry_latent.png    same in the original latent basis (4 axes with the largest variance)
    pca_vs_basis.png         PCA vs original basis: variance per direction, cumulative variance,
                                    PC <-> axis correlation, feature correlations with each basis
    codebook_spacing.png            nearest-code distance per level; within-code spread / feature std
    codebook_features.png           level-0 code centroids for all pairs of features (size ~ usage)
    latent_<pca|tsne|umap>.png      2D projections of encoder latents coloured by features, rank, code
    codebook_quantisation_error.png   |z - z_q| and the residual left after each level
  reconstruction, pre-processed space
    feature_closure.png             input vs reconstructed features, with ratio
    feature_residuals.png           residual distributions, resolution vs the feature value
    feature_correlations.png        feature correlation matrices: input, reconstructed, difference
    resolution_vs_rank.png          IQR68/2*std of every feature vs constituent index
  reconstruction, physical space
    feature_closure_physical.png    same as feature_closure, in physical units
    jet_closure.png                 jet observables + per-jet response (4vec, relvec)
    jet_images.png                  mean jet image: input, reconstructed, difference (4vec, relvec)
    lund_plane.png                  Lund-plane density: input, reconstructed, difference (ktdr)
  training (when a history is given)
    training_history.png            reconstruction, commitment, perplexity, code usage per epoch
  token statistics (--token-stats)
    token_stats.png                 multiplicity, per-position code entropy, code-vs-position map,
                                    usage vs a training reference (if one is given)

Metric conventions
    IQR68         68% IQR interval of the residual (robust resolution)
    bias          median residual (median response - 1 for relative/jet quantities)
    W1            1-Wasserstein distance between input and reconstructed marginals / input std
    perplexity    2^H of the code usage per level (= K for perfectly uniform usage)
    level MI      mutual information between level-0 and level-1 codes / min(H0, H1)
The warning thresholds in FLAG_THRESHOLDS are rules of thumb; tune them to your needs.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import helpers  # noqa: E402
import models_vqvae  # noqa: E402

# ---------------------------------------------------------------------
# Style: fixed categorical order, single-hue sequential, blue<->red diverging
# ---------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TRUE_C, REC_C = SERIES[0], SERIES[1]  # input = blue, reconstructed = orange, everywhere
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CRITICAL = "#d03b3b"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#3987e5", "#f0efec", "#e66767", "#a8302f"])
RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 8,
    "text.color": INK, "font.size": 9, "axes.prop_cycle": cycler(color=SERIES),
}

EPS = 1e-12

# Feature order per input format (same as the helpers datasets)
FEATURE_NAMES = {
    "4vec": [r"$E$", r"$p_{x}$", r"$p_{y}$", r"$p_{z}$"],
    "relvec": [r"$p_{t}$ frac", r"$\Delta\eta$", r"$\Delta\phi$"],
    "ktdr": [r"$\ln(k_{t})$", r"$\ln(1/\Delta R)$"],
}
SCALE_FEATURES = (r"$E$", r"$p_{t}$ frac")  # positive, multi-decade: relative residuals and log axes in physical space
JET_LABELS = {"pt": r"jet $p_T$", "mass": "jet mass", "width": "width", "ptd": r"$p_T^D$"}

FLAG_THRESHOLDS = dict(
    min_utilisation=0.5,          # fraction of codes used per level
    max_top10_share=0.5,          # share of tokens taken by the 10 most used codes
    max_duplicate_fraction=0.05,  # codes sitting on top of another code
    max_level_mi=0.5,             # normalised MI between level 0 and 1
    max_final_rel_residual=0.3,   # |z - z_q| / |z| after the last level
    min_level_gain=0.05,          # relative residual reduction a level must bring
    max_feature_W1=0.05,          # W1/std for constituent features (both spaces)
    max_corr_diff=0.05,           # largest change of a feature-feature correlation
    max_jet_IQR68=0.05,           # per-jet response resolution
    max_jet_bias=0.02,            # |median response - 1|
    max_jet_W1=0.05,              # W1/std for jet observables
    max_image_l1=0.15,            # L1 difference of mean jet image / Lund plane
    max_ood_fraction=0.03,        # vs ~0.01 expected in-distribution (needs a reference)
    max_usage_js=0.05,            # bits (needs a reference)
)


# ---------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------
def _q68(x):
    """median, half-width of central 68% interval"""
    q16, q50, q84 = np.percentile(x, [16, 50, 84])
    return float(q50), float((q84 - q16) / 2)


def _entropy_bits(counts, miller_madow=False):
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    n = counts.sum()
    p = counts / n
    h = float(-(p * np.log2(p)).sum())
    if miller_madow:  # first-order bias correction for sparse histograms
        h += (counts.size - 1) / (2 * n * np.log(2))
    return h


def _js_bits(p, q):
    p, q = np.asarray(p, float) / np.sum(p), np.asarray(q, float) / np.sum(q)
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _W1(t, r):
    return float(wasserstein_distance(t, r) / np.std(t))


def _wrap_phi(phi):
    return np.remainder(phi + np.pi, 2 * np.pi) - np.pi


def _py(o):
    """make metrics JSON-serialisable"""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ---------------------------------------------------------------------
# Collection: run the tokenizer once and keep everything the metrics/plots need
# ---------------------------------------------------------------------
@torch.no_grad()
def collect(tok, X, mask, jet=None, n_jets=None, batch_size=4096, device="cpu"):
    """
    Encode/decode jets and gather everything the diagnostics need (numpy, on CPU).
      X     (J, N, F) pre-processed features (dataset.X)
      mask  (J, N) bool, True = padded (dataset.mask)
      jet   (J, 5) [pt, eta, phi, m, N] (dataset.jet_data), needed for relvec physics
    """
    tok.eval()
    J = len(X) if n_jets is None else min(n_jets, len(X))
    X = torch.as_tensor(np.asarray(X[:J])).float()
    mask = torch.as_tensor(np.asarray(mask[:J])).bool()

    keep = {k: [] for k in ("z", "codes", "rel_res", "qdist", "rec")}
    sq_err = torch.zeros(X.shape[-1])
    for i in range(0, J, batch_size):
        x, m = X[i:i + batch_size].to(device), mask[i:i + batch_size].to(device)
        valid = x[~m]
        z = tok.encoder(valid)
        znorm = z.norm(dim=1).clamp_min(EPS)

        # residual VQ, level by level
        residual, codes, rel = z, [], []
        for vq in tok.vq.levels:
            c = vq.quantize(residual)
            residual = residual - vq.lookup(c)
            codes.append(c)
            rel.append(residual.norm(dim=1) / znorm)
        rec = tok.decoder(z - residual)
        sq_err += ((rec - valid) ** 2).sum(0).cpu()

        keep["z"].append(z.cpu())
        keep["codes"].append(torch.stack(codes, -1).cpu())
        keep["rel_res"].append(torch.stack(rel, -1).cpu())
        keep["qdist"].append(residual.norm(dim=1).cpu())
        keep["rec"].append(rec.cpu())

    real = ~mask
    N = mask.shape[1]
    f_rec = torch.cat(keep["rec"])
    X_rec = torch.full_like(X, float(tok.pad_value))  # reconstructed jets, same padding as the input
    X_rec[real] = f_rec
    return {
        "K": tok.K, "Q": tok.Q, "N": N, "F": X.shape[-1], "n_jets": J,
        "X_true": X.numpy(), "X_rec": X_rec.numpy(), "mask": mask.numpy(),
        "jet": None if jet is None else np.asarray(jet[:J], dtype=np.float64),
        "z": torch.cat(keep["z"]).numpy(), "codes": torch.cat(keep["codes"]).numpy(),
        "rel_res": torch.cat(keep["rel_res"]).numpy(), "qdist": torch.cat(keep["qdist"]).numpy(),
        "f_true": X[real].numpy().astype(np.float64), "f_rec": f_rec.numpy().astype(np.float64),
        "rank": torch.arange(N).expand(J, N)[real].numpy(),
        "jet_idx": torch.arange(J).unsqueeze(1).expand(J, N)[real].numpy(),
        "mult": real.sum(1).numpy(),
        "mse": (sq_err / real.sum()).numpy(),
        "codebooks": [vq.embed.detach().cpu().numpy() for vq in tok.vq.levels],
    }


def compute_reference(col):
    """Code usage and quantisation-error quantiles of a reference (training) sample. Pass the result as
    reference= to later run_diagnostics calls to get the shift / OOD metrics."""
    usage = [np.bincount(col["codes"][:, q], minlength=col["K"]).astype(float) for q in range(col["Q"])]
    q = np.percentile(col["qdist"], [50, 90, 99])
    return {"usage": [(u / u.sum()).tolist() for u in usage],
            "qdist_q50": float(q[0]), "qdist_q90": float(q[1]), "qdist_q99": float(q[2])}


# ---------------------------------------------------------------------
# Physics view
# ---------------------------------------------------------------------
def resolve_preprocess(input_format, preprocess):
    """
    Pre-processing already applied to the input files. Same convention as parse_input in
    helpers_unbinned / helpers_binned: 4vec and relvec inputs are always logstd.
    """
    if input_format in ("4vec", "relvec"):
        return "logstd"
    return preprocess

def _undo(X, input_format, preprocess):
    if not preprocess:
        return X.astype(np.float64)
    return helpers.undo_preprocess(torch.from_numpy(X.copy()), input_format, preprocess).numpy().astype(np.float64)


def _jet_axis(p4, jet_idx, J):
    S = np.stack([np.bincount(jet_idx, weights=p4[:, k], minlength=J) for k in range(4)], -1)
    pt, eta, phi, m, _ = helpers.jet_PtEtaPhiM(S)
    return {"pt": pt[:, 0], "eta": eta[:, 0], "phi": phi[:, 0], "mass": m[:, 0]}


def _coords(p4, jet_idx, axis):
    """per-constituent (d eta, d phi, pt) w.r.t. a given jet axis"""
    cpt = np.hypot(p4[:, 1], p4[:, 2])
    ceta = np.arcsinh(p4[:, 3] / np.maximum(cpt, EPS))
    cphi = np.arctan2(p4[:, 2], p4[:, 1])
    return ceta - axis["eta"][jet_idx], _wrap_phi(cphi - axis["phi"][jet_idx]), cpt


def _jet_observables(p4, jet_idx, J):
    """jet pT, mass, width, pT_D from (M, 4) constituents, each jet w.r.t. its own axis"""
    axis = _jet_axis(p4, jet_idx, J)
    deta, dphi, cpt = _coords(p4, jet_idx, axis)
    sum_pt = np.bincount(jet_idx, weights=cpt, minlength=J)
    width = np.bincount(jet_idx, weights=cpt * np.hypot(deta, dphi), minlength=J) / axis["pt"]
    ptd = np.sqrt(np.bincount(jet_idx, weights=cpt ** 2, minlength=J)) / sum_pt
    return {"pt": axis["pt"], "mass": axis["mass"], "width": width, "ptd": ptd}, axis


def physics_view(col, input_format, preprocess):
    """Physical-space features, plus jet observables / jet image (4vec, relvec) or Lund plane (ktdr)."""
    real = ~col["mask"]
    P_true = _undo(col["X_true"], input_format, preprocess)
    P_rec = _undo(col["X_rec"], input_format, preprocess)
    phys = {"p_true": P_true[real], "p_rec": P_rec[real], "kind": None}
    J, jidx = col["n_jets"], col["jet_idx"]

    if input_format == "ktdr":
        x_t, y_t = phys["p_true"][:, 1], phys["p_true"][:, 0]  # ln 1/dR, ln kt
        x_r, y_r = phys["p_rec"][:, 1], phys["p_rec"][:, 0]
        ex = np.linspace(*np.percentile(x_t, [0.2, 99.8]), 31)
        ey = np.linspace(*np.percentile(y_t, [0.2, 99.8]), 31)
        phys.update(kind="lund", img_true=np.histogram2d(x_t, y_t, bins=[ex, ey])[0] / J,
                    img_rec=np.histogram2d(x_r, y_r, bins=[ex, ey])[0] / J,
                    img_extent=[ex[0], ex[-1], ey[0], ey[-1]], img_labels=(r"$\ln(1/\Delta R)$", r"$\ln k_t$"),
                    img_title="Lund-plane density (emissions per jet per bin)")
        return phys

    # 4-vectors of the real constituents
    if input_format == "relvec":
        p4_t = helpers.make_absolute_constituents(P_true, col["jet"])[real]
        p4_r = helpers.make_absolute_constituents(P_rec, col["jet"])[real]
    else:
        p4_t, p4_r = phys["p_true"], phys["p_rec"]

    obs_t, axis = _jet_observables(p4_t, jidx, J)
    obs_r, _ = _jet_observables(p4_r, jidx, J)
    phys.update(kind="jets", jets_true=obs_t, jets_rec=obs_r)

    # mass shell: E, px, py, pz are quantised independently in 4vec
    if input_format == "4vec":
        for tag, p4 in (("true", p4_t), ("rec", p4_r)):
            m2 = p4[:, 0] ** 2 - (p4[:, 1:] ** 2).sum(1)
            phys[f"m_over_E_{tag}"] = np.sqrt(np.abs(m2)) / np.abs(p4[:, 0])
            phys[f"m2_negative_{tag}"] = float((m2 < 0).mean())

    # mean jet image, both w.r.t. the INPUT jet axis, pT-fraction weighted
    deta_t, dphi_t, pt_t = _coords(p4_t, jidx, axis)
    deta_r, dphi_r, pt_r = _coords(p4_r, jidx, axis)
    R = float(max(0.4, np.percentile(np.hypot(deta_t, dphi_t), 99.5)))
    e = np.linspace(-R, R, 26)  # coarse: finer pixels mostly show bin-migration noise
    jpt = axis["pt"][jidx]
    phys.update(img_true=np.histogram2d(deta_t, dphi_t, bins=[e, e], weights=pt_t / jpt)[0] / J,
                img_rec=np.histogram2d(deta_r, dphi_r, bins=[e, e], weights=pt_r / jpt)[0] / J,
                img_extent=[-R, R, -R, R], img_labels=(r"$\Delta\eta$", r"$\Delta\phi$"),
                img_title=r"mean jet image ($p_T$-fraction weighted)")
    return phys


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def _feature_metrics(t, r, names, relative_scale=False):
    out = {}
    for j, name in enumerate(names):
        tj, rj = t[:, j], r[:, j]
        if relative_scale and name in SCALE_FEATURES:
            med, IQR68 = _q68(rj / tj)
            out[name] = {"type": "relative", "bias": med - 1, "IQR68": IQR68, "W1": _W1(tj, rj)}
        else:
            res = rj - tj
            if name == r"$\Delta\phi$":
                res = _wrap_phi(res)
            b, IQR68 = _q68(res)
            out[name] = {"type": "absolute", "bias": b, "IQR68": IQR68, "IQR68_over_std": IQR68 / np.std(tj),
                         "rms": float(np.sqrt((res ** 2).mean())), "W1": _W1(tj, rj)}
    return out


def compute_metrics(col, names, phys, reference=None, token_stats=False):
    K, Q = col["K"], col["Q"]
    codes = col["codes"]
    m = {"n_jets": col["n_jets"], "n_constituents": int(len(codes)), "K": K, "Q": Q, "features": names}

    # codebook health
    levels, Hs = [], []
    for q in range(Q):
        counts = np.bincount(codes[:, q], minlength=K)
        used = counts > 0
        H = _entropy_bits(counts)
        Hs.append(H)
        E = torch.from_numpy(col["codebooks"][q][used])
        d = torch.cdist(E, E)
        d.fill_diagonal_(float("inf"))
        nn = d.min(1).values.numpy()
        lvl = {"used": int(used.sum()), "utilisation": float(used.mean()), "dead": int((~used).sum()),
               "entropy_bits": H, "perplexity": float(2 ** H), "perplexity_over_K": float(2 ** H / K),
               "top10_share": float(np.sort(counts)[::-1][:10].sum() / counts.sum()),
               "rel_residual_after_mean": float(col["rel_res"][:, q].mean()),
               "nn_distance_median": float(np.median(nn)),
               "duplicate_codes": int((nn < 1e-2 * np.median(E.norm(dim=1).numpy())).sum())}
        if reference is not None:
            lvl["usage_js_vs_train"] = _js_bits(counts, reference["usage"][q])
        levels.append(lvl)
    m["levels"] = levels
    m["bits_per_constituent"] = float(sum(Hs))
    m["max_bits_per_constituent"] = float(Q * np.log2(K))
    if Q >= 2:
        H0 = _entropy_bits(np.bincount(codes[:, 0], minlength=K), True)
        H1 = _entropy_bits(np.bincount(codes[:, 1], minlength=K), True)
        mi = max(0.0, H0 + H1 - _entropy_bits(np.bincount(codes[:, 0] * K + codes[:, 1], minlength=K * K), True))
        m["level01_mi_bits"], m["level01_mi_normalised"] = mi, float(mi / min(H0, H1))

    qd = col["qdist"]
    m["latent_qdist_median"], m["latent_qdist_q99"] = float(np.median(qd)), float(np.percentile(qd, 99))
    if reference is not None:
        m["ood_fraction"] = float((qd > reference["qdist_q99"]).mean())

    # reconstruction, pre-processed space
    m["recon_mse"] = float(col["mse"].mean())  # same quantity as the training loss
    m["preprocessed"] = _feature_metrics(col["f_true"], col["f_rec"], names)
    for j, name in enumerate(names):
        m["preprocessed"][name]["mse"] = float(col["mse"][j])
    m["corr_true"], m["corr_rec"] = np.corrcoef(col["f_true"].T), np.corrcoef(col["f_rec"].T)
    m["corr_max_abs_diff"] = float(np.abs(m["corr_rec"] - m["corr_true"]).max())
    res = col["f_rec"] - col["f_true"]
    sd = col["f_true"].std(0)
    m["IQR68_over_std_vs_rank"] = {name: [_q68(res[col["rank"] == r, j])[1] / sd[j] for r in range(col["N"])
                                          if (col["rank"] == r).sum() > 1] for j, name in enumerate(names)}

    # physical space
    m["physical"] = _feature_metrics(phys["p_true"], phys["p_rec"], names, relative_scale=True)
    if phys["kind"] == "jets":
        m["jet"] = {}
        for key in phys["jets_true"]:
            t, r = phys["jets_true"][key], phys["jets_rec"][key]
            med, IQR68 = _q68(r / t)
            m["jet"][key] = {"response_median": med, "response_IQR68": IQR68, "W1": _W1(t, r)}
        if "m_over_E_true" in phys:
            m["constituent_mass"] = {
                "median_m_over_E_input": float(np.median(phys["m_over_E_true"])),
                "median_m_over_E_reco": float(np.median(phys["m_over_E_rec"])),
                "frac_m2_negative_input": phys["m2_negative_true"], "frac_m2_negative_reco": phys["m2_negative_rec"]}
    it, ir = phys["img_true"], phys["img_rec"]
    m["image_l1"] = float(np.abs(ir - it).sum() / it.sum())

    if token_stats:
        m["tokens"] = token_metrics(col)
    m["flags"] = flags(m)
    return m


def token_metrics(col):
    """Sequence statistics relevant for the next-token model (level-0 codes)."""
    K, N = col["K"], col["N"]
    c0, rank, jet = col["codes"][:, 0], col["rank"], col["jet_idx"]
    same = jet[1:] == jet[:-1]
    prev, nxt = c0[:-1][same], c0[1:][same]
    H_next = _entropy_bits(np.bincount(nxt, minlength=K), True)
    H_joint = _entropy_bits(np.bincount(prev * K + nxt, minlength=K * K), True)
    H_prev = _entropy_bits(np.bincount(prev, minlength=K), True)
    mult = col["mult"]
    return {"tokens_per_jet_mean": float(mult.mean() * col["Q"]), "tokens_per_jet_max": int(mult.max() * col["Q"]),
            "multiplicity_mean": float(mult.mean()), "multiplicity_std": float(mult.std()),
            "level0_entropy_per_position_bits": [_entropy_bits(np.bincount(c0[rank == r], minlength=K), True)
                                                 for r in range(N) if (rank == r).sum() > 1],
            "level0_entropy_bits": _entropy_bits(np.bincount(c0, minlength=K), True),
            "level0_neighbour_mi_bits": float(max(0.0, H_prev + H_next - H_joint)),
            "n_pairs": int(same.sum()),
            "note": "Miller-Madow corrected plug-in entropies; MI is biased high unless n_pairs >> K^2"}


def flags(m):
    T, out = FLAG_THRESHOLDS, []
    for q, lvl in enumerate(m["levels"]):
        if lvl["utilisation"] < T["min_utilisation"]:
            out.append(f"level {q}: only {lvl['utilisation']:.0%} of codes used -> codebook too large for the data, "
                       f"too few optimiser steps (smaller --batch-size), or raise the dead-code restart threshold")
        if lvl["top10_share"] > T["max_top10_share"]:
            out.append(f"level {q}: 10 codes carry {lvl['top10_share']:.0%} of tokens -> partial codebook collapse")
        if lvl["duplicate_codes"] / lvl["used"] > T["max_duplicate_fraction"]:
            out.append(f"level {q}: {lvl['duplicate_codes']} near-duplicate codes -> wasted capacity")
        if q > 0:
            gain = 1 - lvl["rel_residual_after_mean"] / m["levels"][q - 1]["rel_residual_after_mean"]
            if gain < T["min_level_gain"]:
                out.append(f"level {q} reduces the latent residual by only {gain:.1%} -> drop it (shorter sequences)")
        if lvl.get("usage_js_vs_train", 0) > T["max_usage_js"]:
            out.append(f"level {q}: code usage differs from training (JS {lvl['usage_js_vs_train']:.3f} bits)")
    if m["levels"][-1]["rel_residual_after_mean"] > T["max_final_rel_residual"]:
        out.append(f"latent poorly quantised: |z-z_q|/|z| = {m['levels'][-1]['rel_residual_after_mean']:.2f} "
                   f"-> more levels / codes or smaller code_dim")
    if m.get("level01_mi_normalised", 0) > T["max_level_mi"]:
        out.append(f"levels 0 and 1 strongly dependent (normalised MI {m['level01_mi_normalised']:.2f})")
    for space in ("preprocessed", "physical"):
        for key, v in m[space].items():
            if v["W1"] > T["max_feature_W1"]:
                out.append(f"{space} {key}: reconstructed distribution off (W1/std {v['W1']:.3f})")
    if m["corr_max_abs_diff"] > T["max_corr_diff"]:
        out.append(f"feature correlations not preserved (max |deltaρ| = {m['corr_max_abs_diff']:.3f})")
    for key, v in m.get("jet", {}).items():
        if v["response_IQR68"] > T["max_jet_IQR68"]:
            out.append(f"jet {key}: response resolution {v['response_IQR68']:.1%}")
        if abs(v["response_median"] - 1) > T["max_jet_bias"]:
            out.append(f"jet {key}: response biased ({v['response_median'] - 1:+.1%})")
        if v["W1"] > T["max_jet_W1"]:
            out.append(f"jet {key}: reconstructed distribution off (W1/std {v['W1']:.3f})")
    cm = m.get("constituent_mass")
    if cm and cm["median_m_over_E_reco"] > max(3 * cm["median_m_over_E_input"], 0.02):
        out.append(f"reconstructed constituents are off mass shell (median m/E {cm['median_m_over_E_reco']:.3f} vs "
                   f"{cm['median_m_over_E_input']:.3f} input, {cm['frac_m2_negative_reco']:.0%} with m^2 < 0): E and p "
                   f"are quantised independently -> biases jet mass; prefer relvec, or project E onto |p| after decoding")
    if m["image_l1"] > T["max_image_l1"]:
        out.append(f"mean jet image / Lund plane differs (L1 {m['image_l1']:.3f})")
    if m.get("ood_fraction", 0) > T["max_ood_fraction"]:
        out.append(f"{m['ood_fraction']:.1%} of constituents beyond the reference 99th-percentile quantisation "
                   f"error -> this data is out of the tokenizer's training distribution")
    return out


def print_summary(m, outdir=None):
    print(f"== VQ-VAE diagnostics: {m['n_jets']} jets, {m['n_constituents']} constituents ==")
    for q, lv in enumerate(m["levels"]):
        js = f"  usage JS vs train {lv['usage_js_vs_train']:.4f}" if "usage_js_vs_train" in lv else ""
        print(f"  level {q}: used {lv['used']}/{m['K']} ({lv['utilisation']:.0%}), perplexity {lv['perplexity']:.0f} "
              f"({lv['perplexity_over_K']:.0%} of K), top-10 share {lv['top10_share']:.1%}, "
              f"|z-zq|/|z| after {lv['rel_residual_after_mean']:.3f}{js}")
    mi = f", level-0/1 normalised MI {m['level01_mi_normalised']:.3f}" if "level01_mi_normalised" in m else ""
    print(f"  bits/constituent {m['bits_per_constituent']:.1f} of {m['max_bits_per_constituent']:.1f}{mi}")
    lt = m["latent"]
    print(f"  latent: participation ratio {lt['participation_ratio']:.1f} of {lt['code_dim']}; 90% variance in "
          f"{lt['n_pc_90']} PCs vs {lt['n_dims_90']} original axes")
    print(f"  recon MSE {m['recon_mse']:.5f}; max |delta correlation| {m['corr_max_abs_diff']:.4f}")
    print("  pre-processed (IQR68/std, W1/std): " + ", ".join(
        f"{k} {v['IQR68_over_std']:.4f}/{v['W1']:.3f}" for k, v in m["preprocessed"].items()))
    print("  physical (bias, IQR68, W1/std): " + ", ".join(
        f"{k} {v['bias']:+.4f} {v['IQR68']:.4f}{' rel' if v['type'] == 'relative' else ''} {v['W1']:.3f}"
        for k, v in m["physical"].items()))
    if "jet" in m:
        print("  jets (response median ± IQR68, W1/std): " + ", ".join(
            f"{k} {v['response_median']:.3f}±{v['response_IQR68']:.3f} ({v['W1']:.3f})" for k, v in m["jet"].items()))
    if "constituent_mass" in m:
        cm = m["constituent_mass"]
        print(f"  constituent m/E median: input {cm['median_m_over_E_input']:.4f}, reco {cm['median_m_over_E_reco']:.4f}"
              f" (m^2 < 0: {cm['frac_m2_negative_reco']:.1%})")
    print(f"  image L1 difference {m['image_l1']:.3f}")
    if "ood_fraction" in m:
        print(f"  OOD: {m['ood_fraction']:.2%} of constituents beyond reference q99 quantisation error (~1% expected)")
    if "tokens" in m:
        t = m["tokens"]
        print(f"  tokens/jet {t['tokens_per_jet_mean']:.1f} (max {t['tokens_per_jet_max']}), level-0 entropy "
              f"{t['level0_entropy_bits']:.2f} bits, neighbour MI {t['level0_neighbour_mi_bits']:.2f} bits")
    print("  flags: " + ("none" if not m["flags"] else ""))
    for f in m["flags"]:
        print(f"    ! {f}")
    if outdir:
        print(f"  plots + metrics.json -> {outdir}")


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def _save(fig, outdir, name):
    path = Path(outdir) / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _bins(x, n=60, log=False):
    lo, hi = np.percentile(x, [0.1, 99.9])
    if log:
        return np.logspace(np.log10(lo) - 0.05, np.log10(hi) + 0.05, n + 1)
    pad = 0.05 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n + 1)


def _pair_hist(ax, t, r, bins):
    ht, _ = np.histogram(t, bins)
    hr, _ = np.histogram(r, bins)
    ax.stairs(ht, bins, fill=True, color=TRUE_C, alpha=0.18)
    ax.stairs(ht, bins, color=TRUE_C, lw=2, label="input")
    ax.stairs(hr, bins, color=REC_C, lw=2, label="reconstructed")
    return ht, hr


def _ratio(ax, bins, ht, hr, log=False):
    c = np.sqrt(bins[1:] * bins[:-1]) if log else 0.5 * (bins[1:] + bins[:-1])
    ok = (ht > 0) & (hr > 0)
    ratio = hr[ok] / ht[ok]
    ax.axhline(1, color=AXIS, lw=1)
    ax.errorbar(c[ok], ratio, yerr=ratio * np.sqrt(1 / hr[ok] + 1 / ht[ok]), fmt="o", ms=3.5, color=REC_C,
                elinewidth=1)
    ax.set_ylim(0.75, 1.25)


def plot_feature_closure(t, r, names, metrics, outdir, fname, title, log_scale_features=False):
    F = len(names)
    fig = plt.figure(figsize=(3.8 * F, 4.4))
    gs = fig.add_gridspec(2, F, height_ratios=[3, 1], hspace=0.08, wspace=0.28)
    for j, name in enumerate(names):
        ax = fig.add_subplot(gs[0, j])
        axr = fig.add_subplot(gs[1, j], sharex=ax)
        log = log_scale_features and name in SCALE_FEATURES and t[:, j].min() > 0
        bins = _bins(t[:, j], log=log)
        ht, hr = _pair_hist(ax, t[:, j], r[:, j], bins)
        _ratio(axr, bins, ht, hr, log)
        if log:
            ax.set_xscale("log")
        ax.set_title(rf"{name}   $W_{1}/\sigma$ = {metrics[name]['W1']:.3f}")
        ax.tick_params(labelbottom=False)
        axr.set_xlabel(name)
        if j == 0:
            ax.set_ylabel("constituents")
            axr.set_ylabel("reco / input")
            fig.legend(*ax.get_legend_handles_labels(), loc="upper right", ncol=2, bbox_to_anchor=(0.91, 1.0))
    fig.suptitle(title, x=0.08, ha="left", y=1.0)
    return _save(fig, outdir, fname)


def plot_feature_residuals(col, m, names, outdir):
    F = len(names)
    res = col["f_rec"] - col["f_true"]
    fig, axes = plt.subplots(2, F, figsize=(3.8 * F, 6.4), gridspec_kw={"hspace": 0.45, "wspace": 0.3},
                             squeeze=False)
    for j, name in enumerate(names):
        ax, ax2 = axes[0, j], axes[1, j]
        r, t = res[:, j], col["f_true"][:, j]
        lim = np.percentile(np.abs(r), 99.5) * 1.1
        ax.hist(r, np.linspace(-lim, lim, 81), histtype="stepfilled", color=TRUE_C, alpha=0.85)
        ax.set_yscale("log")
        ax.axvline(0, color=AXIS, lw=1)
        v = m["preprocessed"][name]
        ax.set_title(f"reco − input: {name}")
        ax.text(0.03, 0.95, rf"bias {v['bias']:+.4f}\n$IQR_{68}/2$ {v['IQR68']:.4f}", transform=ax.transAxes,
                va="top", color=INK2, fontsize=8)
        edges = np.unique(np.percentile(t, np.linspace(0, 100, 13)))
        which = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)
        prof = [_q68(r[which == b])[1] for b in range(len(edges) - 1)]
        ax2.plot(0.5 * (edges[1:] + edges[:-1]), prof, "-o", color=TRUE_C, ms=5)
        ax2.set_xlabel(f"input {name}")
        ax2.set_title(f"resolution of {name} vs its value")
        ax2.set_ylim(bottom=0)
        if j == 0:
            ax.set_ylabel("constituents")
            ax2.set_ylabel(r"$IQR_{68}/2$")
    fig.suptitle("Residuals in pre-processed space (the space the loss is computed in)", x=0.08, ha="left", y=0.99)
    return _save(fig, outdir, "feature_residuals.png")


def plot_feature_correlations(m, names, outdir):
    ct, cr = m["corr_true"], m["corr_rec"]
    d = cr - ct
    v = max(np.abs(d).max(), 1e-3)
    F = len(names)
    fig, axes = plt.subplots(1, 3, figsize=(4.4 * 3 + 1.5, 4.2), gridspec_kw={"wspace": 0.45})
    panels = ((ct, "input", TwoSlopeNorm(0, -1, 1)), (cr, "reconstructed", TwoSlopeNorm(0, -1, 1)),
              (d, fr"reco − input (max $|\Delta\rho|$ = {m['corr_max_abs_diff']:.3f})", TwoSlopeNorm(0, -v, v)))
    for ax, (mat, title, norm) in zip(axes, panels):
        im = ax.imshow(mat, cmap=DIV, norm=norm)
        for a in range(F):
            for b in range(F):
                ax.text(b, a, f"{mat[a, b]:+.2f}" if mat is d else f"{mat[a, b]:.2f}", ha="center", va="center",
                        fontsize=8, color=INK)
        ax.set_xticks(range(F), names, rotation=30)
        ax.set_yticks(range(F), names)
        ax.set_title(f"correlation: {title}")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, outdir, "feature_correlations.png")


def plot_resolution_vs_rank(m, names, outdir):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for j, name in enumerate(names):
        y = m["IQR68_over_std_vs_rank"][name]
        ax.plot(np.arange(1, len(y) + 1), y, "-o", ms=5, color=SERIES[j % 8], label=name)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("constituent index (stored order)")
    ax.set_ylabel(r"$IQR_{68}/\sigma_{feature}$")
    ax.set_title("Reconstruction resolution vs constituent index (pre-processed space)")
    ax.legend()
    return _save(fig, outdir, "resolution_vs_rank.png")


def plot_jet_closure(phys, m, outdir):
    keys = list(phys["jets_true"])
    fig, axes = plt.subplots(2, len(keys), figsize=(3.8 * len(keys), 6.4), gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    for j, key in enumerate(keys):
        t, r = phys["jets_true"][key], phys["jets_rec"][key]
        ax, ax2 = axes[0, j], axes[1, j]
        _pair_hist(ax, t, r, _bins(t, 50))
        v = m["jet"][key]
        ax.set_title(rf"{JET_LABELS[key]}   $W_{1}/\sigma$ = {v['W1']:.3f}")
        ax.set_xlabel(JET_LABELS[key])
        half = max(4 * v["response_IQR68"], 0.01)
        ax2.hist(r / t, np.linspace(v["response_median"] - half, v["response_median"] + half, 61),
                 histtype="stepfilled", color=TRUE_C, alpha=0.85)
        ax2.axvline(1, color=INK2, lw=1, ls="--")
        ax2.set_title(f"response {JET_LABELS[key]}")
        ax2.set_xlabel("reco / input (per jet)")
        ax2.text(0.03, 0.95, rf"median {v['response_median']:.4f}\n$IQR_{68}/2$ {v['response_IQR68']:.4f}",
                 transform=ax2.transAxes, va="top", color=INK2, fontsize=8)
        if j == 0:
            ax.set_ylabel("jets")
            ax2.set_ylabel("jets")
            fig.legend(*ax.get_legend_handles_labels(), loc="upper right", ncol=2, bbox_to_anchor=(0.91, 0.99))
    return _save(fig, outdir, "jet_closure.png")


def plot_density_images(phys, m, outdir, fname):
    it, ir, ext = phys["img_true"], phys["img_rec"], phys["img_extent"]
    xl, yl = phys["img_labels"]
    vmin = max(it[it > 0].min(), it.max() * 1e-4)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"wspace": 0.35})
    for ax, img, title in ((axes[0], it, "input"), (axes[1], ir, "reconstructed")):
        im = ax.imshow(np.clip(img.T, vmin, None), origin="lower", extent=ext, cmap=SEQ, aspect="auto",
                       norm=LogNorm(vmin=vmin, vmax=it.max()))
        ax.set_title(f"{phys['img_title']}: {title}", fontsize=9)
        fig.colorbar(im, ax=ax, label="per jet per bin")
    d = (ir - it) / it.max()
    v = np.abs(d).max()
    im = axes[2].imshow(d.T, origin="lower", extent=ext, cmap=DIV, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
    axes[2].set_title(f"(reco − input) / max   L1 = {m['image_l1']:.3f}")
    fig.colorbar(im, ax=axes[2], label="relative difference")
    for ax in axes:
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(False)
    return _save(fig, outdir, fname)


def plot_codebook_usage(col, m, outdir):
    K, Q = col["K"], col["Q"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"wspace": 0.3, "width_ratios": [1.6, 1]})
    for q in range(Q):
        counts = np.sort(np.bincount(col["codes"][:, q], minlength=K))[::-1]
        frac = counts / counts.sum()
        a1.plot(np.arange(1, K + 1), np.where(frac > 0, frac, np.nan), color=SERIES[q % 8],
                label=f"level {q}  (perplexity {m['levels'][q]['perplexity']:.0f})")
    a1.axhline(1 / K, color=MUTED, lw=1, ls="--", label="uniform usage")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("code rank")
    a1.set_ylabel("fraction of tokens")
    a1.set_title("Code usage, rank-frequency (flat = healthy, steep head = collapse)")
    a1.legend()
    x, w = np.arange(Q), 0.36
    a2.bar(x - w / 2, [lv["utilisation"] for lv in m["levels"]], w * 0.92, color=SERIES[0], label="codes used / K")
    a2.bar(x + w / 2, [lv["perplexity_over_K"] for lv in m["levels"]], w * 0.92, color=SERIES[1],
           label="perplexity / K")
    a2.set_xticks(x, [f"level {q}" for q in range(Q)])
    a2.set_ylim(0, 1.05)
    a2.set_title("Codebook utilisation")
    a2.legend(loc="lower right")
    return _save(fig, outdir, "codebook_usage.png")


def plot_level_cooccurrence(col, m, outdir):
    K = col["K"]
    c0, c1 = col["codes"][:, 0], col["codes"][:, 1]
    r0, r1 = np.empty(K, int), np.empty(K, int)
    r0[np.argsort(-np.bincount(c0, minlength=K))] = np.arange(K)
    r1[np.argsort(-np.bincount(c1, minlength=K))] = np.arange(K)
    joint = np.bincount(r0[c0] * K + r1[c1], minlength=K * K).reshape(K, K)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(np.ma.masked_equal(joint, 0).T, origin="lower", cmap=SEQ, norm=LogNorm(vmin=1),
                   aspect="auto", interpolation="nearest")
    ax.set_xlabel("level-0 code (sorted by usage)")
    ax.set_ylabel("level-1 code (sorted by usage)")
    ax.set_title(f"Level co-occurrence   normalised MI = {m['level01_mi_normalised']:.3f}\n"
                 "(uniform smear = independent refinement; structure = redundancy)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="constituents")
    return _save(fig, outdir, "level_cooccurrence.png")


def _subsample(n, k, seed=0):
    return np.sort(np.random.default_rng(seed).choice(n, size=min(n, k), replace=False))


# ---------------------------------------------------------------------
# Latent space / codebook geometry
# ---------------------------------------------------------------------
def _corr(a, b):
    """Pearson correlation between the columns of a (M, A) and b (M, B) -> (A, B)"""
    a = (a - a.mean(0)) / np.maximum(a.std(0), EPS)
    b = (b - b.mean(0)) / np.maximum(b.std(0), EPS)
    return a.T @ b / len(a)


def latent_analysis(col, max_points=20000):
    """
    PCA of the encoder latents, shared by the geometry plots and the latent metrics.
    The original latent axes are ordered by their variance, so 'axis 1' is the most used one.
    """
    idx = _subsample(len(col["z"]), max_points)
    z = col["z"][idx]
    f = col["f_true"][idx]
    pca = PCA().fit(z)
    pz = pca.transform(z)
    dim_var = z.var(0)
    order = np.argsort(-dim_var)
    return {
        "idx": idx, "z": z, "pz": pz, "pca": pca, "order_dims": order,
        "pca_var_frac": pca.explained_variance_ratio_, "dim_var_frac": dim_var[order] / dim_var.sum(),
        "corr_pc_dim": _corr(pz, z[:, order]),    # (PC, original axis)
        "corr_feat_pc": _corr(f, pz),             # (feature, PC)
        "corr_feat_dim": _corr(f, z[:, order]),   # (feature, original axis)
    }


def latent_metrics(lat, names):
    """How compact the latent space is in the PCA vs the original basis, and how physical each basis is."""
    cum_pca, cum_dim = np.cumsum(lat["pca_var_frac"]), np.cumsum(lat["dim_var_frac"])
    ev = lat["pca"].explained_variance_
    return {
        "code_dim": int(len(ev)),
        "pca_var_frac": lat["pca_var_frac"], "dim_var_frac": lat["dim_var_frac"],
        "n_pc_90": int(np.searchsorted(cum_pca, 0.90) + 1), "n_pc_99": int(np.searchsorted(cum_pca, 0.99) + 1),
        "n_dims_90": int(np.searchsorted(cum_dim, 0.90) + 1), "n_dims_99": int(np.searchsorted(cum_dim, 0.99) + 1),
        "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),  # effective number of latent dimensions
        "best_abs_corr_pc": {n: float(np.abs(lat["corr_feat_pc"][j]).max()) for j, n in enumerate(names)},
        "best_abs_corr_dim": {n: float(np.abs(lat["corr_feat_dim"][j]).max()) for j, n in enumerate(names)},
    }


def _code_stats(col):
    """Level-0 code usage, centroid in feature space and within-code spread / feature std."""
    K, F = col["K"], col["F"]
    c0, f = col["codes"][:, 0], col["f_true"]
    n = np.bincount(c0, minlength=K).astype(float)
    used = n > 0
    mean = np.stack([np.bincount(c0, weights=f[:, j], minlength=K)[used] / n[used] for j in range(F)], -1)
    var = np.stack([np.bincount(c0, weights=f[:, j] ** 2, minlength=K)[used] / n[used] for j in range(F)], -1) - mean ** 2
    return {"n": n, "used": used, "mean": mean, "spread": np.sqrt(np.clip(var, 0, None)) / f.std(0)}


def _usage_size(counts, smax=50):
    return 4 + smax * np.sqrt(counts / counts.max())


def _corner(n, size=2.7):
    """Lower-triangle grid for all pairs of n coordinates: axes[r, c] shows x = coordinate c, y = coordinate r+1."""
    fig, axes = plt.subplots(n - 1, n - 1, figsize=(size * (n - 1) + 1.2, size * (n - 1)), squeeze=False,
                             gridspec_kw={"hspace": 0.3, "wspace": 0.3})
    for r in range(n - 1):
        for c in range(n - 1):
            if c > r:
                axes[r, c].axis("off")
    return fig, axes


def _latent_corner(lat_xy, code_xy, counts, labels, title, outdir, fname):
    """Corner plot of latents (gray) with the level-0 codebook (size ~ usage, unused = x)."""
    n = len(labels)
    used = counts > 0
    fig, axes = _corner(n)
    for r in range(n - 1):
        for c in range(r + 1):
            ax, xi, yi = axes[r, c], c, r + 1
            ax.scatter(lat_xy[:, xi], lat_xy[:, yi], s=3, color=MUTED, alpha=0.3, lw=0, label="encoder latents")
            ax.scatter(code_xy[used, xi], code_xy[used, yi], s=_usage_size(counts[used], 40), color=SERIES[0],
                       edgecolor=SURFACE, lw=0.4, alpha=0.9, label="level-0 codes (size ∝ usage)")
            ax.scatter(code_xy[~used, xi], code_xy[~used, yi], s=16, marker="x", color=CRITICAL, lw=1.0,
                       label="unused codes")
            if r == n - 2:
                ax.set_xlabel(labels[xi])
            if c == 0:
                ax.set_ylabel(labels[yi])
    if n > 2:
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.9, 0.88),
                   markerscale=1.5)
    else:
        axes[0, 0].legend(loc="best")
    fig.suptitle(title, x=0.06, ha="left", y=0.94)
    return _save(fig, outdir, fname)


def plot_codebook_geometry(col, lat, outdir, n_axes=4):
    """
    Latent space with the level-0 codebook, all pairs of the first n_axes coordinates:
      codebook_geometry.png          principal components of the latents
      codebook_geometry_latent.png   original latent axes (the n_axes with the largest variance)
    """
    K = col["K"]
    n_axes = min(n_axes, lat["z"].shape[1])
    counts = np.bincount(col["codes"][:, 0], minlength=K)
    E = col["codebooks"][0]

    ev = lat["pca_var_frac"]
    labels = [f"PC{k + 1} ({ev[k]:.0%})" for k in range(n_axes)]
    p1 = _latent_corner(lat["pz"][:, :n_axes], lat["pca"].transform(E)[:, :n_axes], counts, labels,
                        f"Latent space with the level-0 codebook: first {n_axes} principal components "
                        f"(% = variance fraction)", outdir, "codebook_geometry.png")

    dims = lat["order_dims"][:n_axes]
    labels = [f"$z_{{{d}}}$ ({lat['dim_var_frac'][k]:.0%})" for k, d in enumerate(dims)]
    p2 = _latent_corner(lat["z"][:, dims], E[:, dims], counts, labels,
                        f"Latent space with the level-0 codebook: the {n_axes} original latent axes with the "
                        f"largest variance (% = variance fraction)", outdir, "codebook_geometry_latent.png")
    return p1, p2


def plot_pca_vs_basis(col, lat, lm, names, outdir):
    """
    PCA vs original latent basis:
      variance per component and cumulative (how compact each basis is),
      correlation PC <-> original axis, and of the input features with each basis (how physical it is).
    """
    D = lat["z"].shape[1]
    k = np.arange(1, D + 1)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), gridspec_kw={"hspace": 0.45, "wspace": 0.35})

    a = axes[0, 0]
    a.plot(k, lat["pca_var_frac"], "-o", ms=5, color=SERIES[0], label="principal components")
    a.plot(k, lat["dim_var_frac"], "-o", ms=5, color=SERIES[1], label="original axes (sorted)")
    a.set_yscale("log")
    a.set_xlabel("component / axis (by variance)")
    a.set_ylabel("variance fraction")
    a.set_title(f"Variance per direction (participation ratio {lm['participation_ratio']:.1f} of {D})")
    a.legend()

    a = axes[0, 1]
    a.plot(k, np.cumsum(lat["pca_var_frac"]), "-o", ms=5, color=SERIES[0],
           label=f"principal components (90%: {lm['n_pc_90']}, 99%: {lm['n_pc_99']})")
    a.plot(k, np.cumsum(lat["dim_var_frac"]), "-o", ms=5, color=SERIES[1],
           label=f"original axes (90%: {lm['n_dims_90']}, 99%: {lm['n_dims_99']})")
    for lvl in (0.9, 0.99):
        a.axhline(lvl, color=MUTED, ls="--", lw=1)
    a.set_ylim(0, 1.02)
    a.set_xlabel("number of components / axes")
    a.set_ylabel("cumulative variance fraction")
    a.set_title("How many directions the latent space really uses")
    a.legend(loc="lower right")

    a = axes[0, 2]
    im = a.imshow(lat["corr_pc_dim"], cmap=DIV, norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
    a.set_xticks(range(D), [f"$z_{{{d}}}$" for d in lat["order_dims"]], rotation=90, fontsize=7)
    a.set_yticks(range(D), [f"PC{i + 1}" for i in range(D)], fontsize=7)
    a.set_xlabel("original latent axis (by variance)")
    a.set_title("Correlation PC vs original axis\n(diagonal = PCA ≈ original basis)")
    a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.046, label="correlation")

    for a, mat, xl, ticks in ((axes[1, 0], lat["corr_feat_pc"], "principal component",
                               [f"PC{i + 1}" for i in range(D)]),
                              (axes[1, 1], lat["corr_feat_dim"], "original latent axis (by variance)",
                               [f"$z_{{{d}}}$" for d in lat["order_dims"]])):
        im = a.imshow(mat, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                a.text(c, r, f"{mat[r, c]:+.1f}", ha="center", va="center", fontsize=6, color=INK)
        a.set_xticks(range(D), ticks, rotation=90, fontsize=7)
        a.set_yticks(range(len(names)), names)
        a.set_xlabel(xl)
        a.set_title(f"Correlation of input features with each {xl.split(' (')[0]}")
        a.grid(False)
        fig.colorbar(im, ax=a, fraction=0.046, label="correlation")

    a = axes[1, 2]
    x, w = np.arange(len(names)), 0.36
    a.bar(x - w / 2, [lm["best_abs_corr_pc"][n] for n in names], w * 0.92, color=SERIES[0], label="best PC")
    a.bar(x + w / 2, [lm["best_abs_corr_dim"][n] for n in names], w * 0.92, color=SERIES[1],
          label="best original axis")
    a.axhline(1, color=AXIS, lw=1)
    a.set_xticks(x, names)
    a.set_ylim(0, 1.2)
    a.set_ylabel("max |correlation| with a single direction")
    a.set_title("How well one direction captures each feature\n(high = disentangled in that basis)")
    a.legend(loc="upper center", ncol=2)
    return _save(fig, outdir, "pca_vs_basis.png")


def plot_codebook_spacing(col, names, outdir):
    """Distance between neighbouring codes per level, and within-code spread of each feature (level 0)."""
    K, Q = col["K"], col["Q"]
    st = _code_stats(col)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"wspace": 0.3})
    for q in range(Q):
        E = torch.from_numpy(col["codebooks"][q][np.bincount(col["codes"][:, q], minlength=K) > 0])
        d = torch.cdist(E, E)
        d.fill_diagonal_(float("inf"))
        nn = d.min(1).values.numpy()
        a1.hist(nn, np.logspace(np.log10(nn.min() + EPS), np.log10(nn.max()), 50), histtype="step",
                lw=2, color=SERIES[q % 8], label=f"level {q}")
    a1.set_xscale("log")
    a1.set_xlabel("distance to nearest other code")
    a1.set_ylabel("codes")
    a1.set_title("Codebook spacing (spike at small distance = duplicates)")
    a1.legend()
    hi = np.percentile(st["spread"], 99.5) * 1.05
    for j, name in enumerate(names):
        a2.hist(st["spread"][:, j], np.linspace(0, hi, 50), histtype="step", lw=2, color=SERIES[j % 8], label=name)
    a2.set_xlabel(r"within-code $\sigma$ / feature $\sigma$")
    a2.set_ylabel("level-0 codes")
    a2.set_title("How wide each code is, level 0 only (later levels refine)")
    a2.legend()
    return _save(fig, outdir, "codebook_spacing.png")


def plot_codebook_features(col, names, outdir, max_points=20000):
    """Level-0 code centroids in feature space, all pairs of features, size ~ usage, over the input constituents."""
    st = _code_stats(col)
    idx = _subsample(len(col["f_true"]), max_points, seed=2)
    f = col["f_true"][idx]
    F = len(names)
    fig, axes = _corner(F)
    for r in range(F - 1):
        for c in range(r + 1):
            ax, xi, yi = axes[r, c], c, r + 1
            ax.scatter(f[:, xi], f[:, yi], s=3, color=MUTED, alpha=0.3, lw=0, label="input constituents")
            ax.scatter(st["mean"][:, xi], st["mean"][:, yi], s=_usage_size(st["n"][st["used"]], 40), color=SERIES[0],
                       edgecolor=SURFACE, lw=0.4, alpha=0.85, label="level-0 code centroids (size ∝ usage)")
            if r == F - 2:
                ax.set_xlabel(f"{names[xi]} (pre-processed)")
            if c == 0:
                ax.set_ylabel(f"{names[yi]} (pre-processed)")
    if F > 2:
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.9, 0.88),
                   markerscale=1.5)
    else:
        axes[0, 0].legend(loc="best")
    fig.suptitle("Level-0 code centroids in feature space (centroid = mean input of the constituents in the code)",
                 x=0.06, ha="left", y=0.94)
    return _save(fig, outdir, "codebook_features.png")


def _project(X, method):
    if method == "pca":
        return PCA(2).fit_transform(X)
    if method == "tsne":
        return TSNE(2, perplexity=30.0, init="pca", learning_rate="auto", random_state=0).fit_transform(X)
    return UMAP(n_components=2, random_state=0).fit_transform(X)


def plot_latent_projection(col, names, outdir, method, max_points=5000):
    idx = _subsample(len(col["z"]), max_points, seed=1)
    P = _project(col["z"][idx], method)
    f = col["f_true"][idx]
    colourings = [(f[:, j], f"{names[j]} (pre-processed)") for j in range(min(col["F"], 3))]
    colourings.append((col["rank"][idx] + 1, "constituent index"))
    n = len(colourings)
    fig, axes = plt.subplots(1, n, figsize=(4.7 * n, 4.4), gridspec_kw={"wspace": 0.25})
    for ax, (c, label) in zip(axes, colourings):
        lo, hi = np.percentile(c, [1, 99])
        sc = ax.scatter(P[:, 0], P[:, 1], c=c, s=3, cmap=SEQ, vmin=lo, vmax=hi, lw=0)
        fig.colorbar(sc, ax=ax, label=label)
        ax.set_title(f"coloured by {label}")
    fig.suptitle(f"Encoder latents, {method.upper()} projection ({len(idx)} constituents). Smooth gradients = "
                 "organised latent space; codes should tile it without large gaps", x=0.06, ha="left", y=1.02)

    fig2, ax = plt.subplots(1, 1, figsize=(4.7, 4.4), gridspec_kw={"wspace": 0.25})
    c0 = col["codes"][idx, 0]
    top = np.argsort(-np.bincount(c0))[:8]
    other = ~np.isin(c0, top)
    ax.scatter(P[other, 0], P[other, 1], s=3, color=MUTED, alpha=0.25, lw=0, label="other codes")
    for k, code in enumerate(top):
        sel = c0 == code
        ax.scatter(P[sel, 0], P[sel, 1], s=6, color=SERIES[k], lw=0, label=f"code {code} ({sel.mean():.1%})")
    ax.set_title("8 most used level-0 codes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=7, markerscale=2.5)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    #fig2.suptitle(f"Encoder latents, {method.upper()} projection ({len(idx)} constituents). Smooth gradients = "
    #             "organised latent space; codes should tile it without large gaps", x=0.06, ha="left", y=1.02)
    return _save(fig, outdir, f"latent_{method}_features.png"), _save(fig2, outdir, f"latent_{method}.png")


def plot_quantisation_error(col, m, outdir, reference=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"wspace": 0.3})
    qd = col["qdist"]
    a1.hist(qd, np.logspace(np.log10(np.percentile(qd, 0.1)), np.log10(qd.max()), 70), histtype="stepfilled",
            color=TRUE_C, alpha=0.85, label="this data")
    if reference is not None:
        a1.axvline(reference["qdist_q99"], color=INK2, ls="--", lw=1.2, label="reference q99")
        a1.text(0.97, 0.95, f"beyond reference q99: {m['ood_fraction']:.2%}\n(~1% expected)",
                transform=a1.transAxes, ha="right", va="top", color=INK2, fontsize=8)
    a1.set_xscale("log")
    a1.set_xlabel(r"latent quantisation error $|z - z_q|$")
    a1.set_ylabel("constituents")
    a1.set_title("Quantisation error (long tail = poorly covered region / OOD)")
    a1.legend(loc="upper left")
    rr = col["rel_res"]
    bins = np.logspace(np.log10(np.percentile(rr, 0.1)), np.log10(rr.max()), 70)
    for q in range(col["Q"]):
        a2.hist(rr[:, q], bins, histtype="step", lw=2, color=SERIES[q % 8],
                label=f"after level {q} (mean {m['levels'][q]['rel_residual_after_mean']:.3f})")
    a2.set_xscale("log")
    a2.set_xlabel(r"remaining residual $|z - \sum_{\leq q} e_q| / |z|$")
    a2.set_title("What each RVQ level removes (curves should step left)")
    a2.legend()
    return _save(fig, outdir, "codebook_quantisation_error.png")


def plot_training_history(history, outdir):
    """history: dict of per-epoch lists as stored by train_vqvae.py"""
    ep = np.arange(1, len(history["train_reco"]) + 1)
    perp = np.asarray(history["perplexity"])   # (epochs, Q)
    used = np.asarray(history["codes_used"])   # (epochs, Q)
    K = history["K"][-1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw={"hspace": 0.4, "wspace": 0.28})
    for ax, what, title in ((axes[0, 0], "reco", "Reconstruction MSE"), (axes[0, 1], "commit", "Commitment loss")):
        ax.plot(ep, history[f"train_{what}"], color=SERIES[0], label="train")
        ax.plot(ep, history[f"test_{what}"], color=SERIES[1], label="test")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend()
    for q in range(perp.shape[1]):
        axes[1, 0].plot(ep, perp[:, q], color=SERIES[q % 8], label=f"level {q}")
        axes[1, 1].plot(ep, used[:, q] / K, color=SERIES[q % 8], label=f"level {q}")
    axes[1, 0].axhline(K, color=MUTED, ls="--", lw=1, label="K")
    axes[1, 0].set_title("Test perplexity")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Fraction of codes used (test)")
    for ax in axes.flat:
        ax.set_xlabel("epoch")
    axes[1, 0].legend()
    axes[1, 1].legend()
    return _save(fig, outdir, "training_history.png")


def plot_token_stats(col, m, outdir, reference=None):
    K, N = col["K"], col["N"]
    t = m["tokens"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    a = axes[0, 0]
    mult = col["mult"]
    a.hist(mult, np.arange(mult.min() - 0.5, mult.max() + 1.5), histtype="stepfilled", color=TRUE_C, alpha=0.85)
    a.set_xlabel("constituents per jet")
    a.set_ylabel("jets")
    a.set_title(f"Multiplicity (tokens/jet = {col['Q']} × this; mean {t['tokens_per_jet_mean']:.1f})")
    a = axes[0, 1]
    positions = [r for r in range(N) if (col["rank"] == r).sum() > 1]
    for q in range(col["Q"]):
        cq = col["codes"][:, q]
        Hq = [_entropy_bits(np.bincount(cq[col["rank"] == r], minlength=K), True) for r in positions]
        a.plot(np.array(positions) + 1, Hq, "-o", ms=5, color=SERIES[q % 8], label=f"level {q}")
    a.axhline(np.log2(K), color=MUTED, ls="--", lw=1, label=r"$\log_2 K$")
    a.set_xlabel("constituent index")
    a.set_ylabel("bits")
    a.set_ylim(bottom=0)
    a.set_title("Code entropy per position (low = predictable from position)")
    a.legend()
    a = axes[1, 0]
    c0 = col["codes"][:, 0]
    top = np.argsort(-np.bincount(c0, minlength=K))[:64]
    H = np.zeros((len(top), N))
    for r in positions:
        sel = col["rank"] == r
        H[:, r] = np.bincount(c0[sel], minlength=K)[top] / sel.sum()
    im = a.imshow(H, aspect="auto", origin="lower", cmap=SEQ, extent=[0.5, N + 0.5, -0.5, len(top) - 0.5],
                  interpolation="nearest")
    fig.colorbar(im, ax=a, label="fraction of constituents at this position")
    a.set_xlabel("constituent index")
    a.set_ylabel("level-0 code (top 64 by usage)")
    a.set_title("Code vs position")
    a.grid(False)
    a = axes[1, 1]
    if reference is None:
        a.set_title("Usage vs training")
        a.text(0.5, 0.5, "no reference given", ha="center", va="center", transform=a.transAxes, color=INK2)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    else:
        for q in range(col["Q"]):
            u = np.bincount(col["codes"][:, q], minlength=K) / len(c0)
            ref = np.asarray(reference["usage"][q])
            ok = (u > 0) & (ref > 0)
            a.scatter(ref[ok], u[ok], s=10, color=SERIES[q % 8], lw=0, alpha=0.8,
                      label=f"level {q} (JS {m['levels'][q]['usage_js_vs_train']:.4f} bits)")
        lo = 0.5 / len(c0)
        a.plot([lo, 1], [lo, 1], color=MUTED, lw=1, ls="--")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("code frequency, training reference")
        a.set_ylabel("code frequency, this file")
        a.set_title("Code usage vs training (on diagonal = same distribution)")
        a.legend()
    return _save(fig, outdir, "token_stats.png")


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def run_diagnostics(col, outdir, input_format, preprocess=None, history=None, reference=None, token_stats=False,
                    plots=True, projections=("pca", "tsne"), max_points=5000, feature_keys=None):
    """Compute metrics (always) and plots (if plots=True) from a collect() result.
    Returns the metrics dict; writes <outdir>/metrics.json. feature_keys optionally overrides the feature names."""
    t0 = time.time()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = list(feature_keys) if feature_keys else FEATURE_NAMES[input_format]
    preprocess = resolve_preprocess(input_format, preprocess)  # only used to undo it for the physical-space plots
    phys = physics_view(col, input_format, preprocess)
    m = compute_metrics(col, names, phys, reference, token_stats=token_stats)
    lat = latent_analysis(col)
    m["latent"] = latent_metrics(lat, names)
    m["input_format"], m["preprocess"] = input_format, preprocess
    if plots:
        with plt.rc_context(RC):
            paths = [
                plot_feature_closure(col["f_true"], col["f_rec"], names, m["preprocessed"], outdir,
                                     "feature_closure.png", "Constituent features, pre-processed space"),
                plot_feature_residuals(col, m, names, outdir),
                plot_feature_correlations(m, names, outdir),
                plot_resolution_vs_rank(m, names, outdir),
                plot_codebook_usage(col, m, outdir),
                *plot_codebook_geometry(col, lat, outdir),
                plot_pca_vs_basis(col, lat, m["latent"], names, outdir),
                plot_codebook_spacing(col, names, outdir),
                plot_codebook_features(col, names, outdir),
                plot_quantisation_error(col, m, outdir, reference),
                plot_feature_closure(phys["p_true"], phys["p_rec"], names, m["physical"], outdir,
                                     "feature_closure_physical.png",
                                     f"Constituent features, physical space (inverse of '{preprocess}')",
                                     log_scale_features=True),
                plot_density_images(phys, m, outdir, "lund_plane.png" if phys["kind"] == "lund" else "jet_images.png"),
            ]
            if col["Q"] >= 2:
                paths.append(plot_level_cooccurrence(col, m, outdir))
            if phys["kind"] == "jets":
                paths.append(plot_jet_closure(phys, m, outdir))
            if history:
                paths.append(plot_training_history(history, outdir))
            if token_stats:
                paths.append(plot_token_stats(col, m, outdir, reference))
            for method in projections:
                paths.extend(plot_latent_projection(col, names, outdir, method, max_points))
        m["plots"] = paths
    m["diagnostics_seconds"] = time.time() - t0
    with open(outdir / "metrics.json", "w") as f:
        json.dump(_py(m), f, indent=2)
    print_summary(m, outdir)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True, help="tokenizer checkpoint from train_vqvae.py")
    p.add_argument("--data", required=True, help="input .h5 in the training format")
    p.add_argument("--out", required=True, help="output directory for plots and metrics.json")
    p.add_argument("--n-jets", type=int, default=20000, help="jets to analyse (first N in the file)")
    p.add_argument("--max-points", type=int, default=5000, help="constituents in t-SNE/UMAP/PCA projections")
    p.add_argument("--projections", nargs="*", default=["pca", "tsne", "umap"], choices=["pca", "tsne", "umap"])
    p.add_argument("--token-stats", action="store_true", help="also sequence/token statistics")
    p.add_argument("--no-plots", action="store_true", help="metrics only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    #Load the tokenizer, and the data exactly as during training
    model, train_args = models_vqvae.load_tokenizer_checkpoint(args.tokenizer, args.device)
    history = torch.load(args.tokenizer, map_location="cpu").get("training_history")
    dataset = helpers.get_loader(args.data, train_args).dataset

    col = collect(model, dataset.X, dataset.mask, dataset.jet_data, n_jets=args.n_jets, device=args.device)
    run_diagnostics(col, args.out, train_args["input_format"], train_args["preprocess"], history=history,
                    token_stats=args.token_stats, plots=not args.no_plots, projections=args.projections,
                    max_points=args.max_points)


if __name__ == "__main__":
    main()
