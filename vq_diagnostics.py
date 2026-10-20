"""
Diagnostics for the constituent VQ-VAE tokenizer (feature-agnostic version).

Works on the pre-processed per-constituent features stored in the input .h5, for any
--input_format (4vec, relvec, ktdr, or anything else given with --feature-keys). Two views:

  pre-processed space (always)    what the VQ-VAE sees and is trained on
  physical space (when possible)  features mapped back with the inverse of helpers.preprocess
                                  (same formulas and constant tables as helpers.undo_preprocess,
                                  applied to real constituents only), plus per-format physics:
                                    4vec    jet pT, mass, girth, pT_D closure + mean jet image
                                    relvec  |sum pT|/pT_J, m/pT_J, girth, pT_D (massless constituents,
                                            jet frame) + mean jet image
                                    ktdr    Lund-plane density (ln 1/dR, ln kt)

Called by train_vqvae.py at the end of training (on the test file), or standalone; the data and
pre-processing settings are read from the checkpoint and can be overridden:

    python vq_diagnostics.py --tokenizer tokenizer.pt --data test.h5 --out diag_test [--token-stats]

Outputs in <out>/ (metrics.json + PNGs):
  codebook health
    codebook_usage.png              rank-frequency of codes per RVQ level; utilisation, perplexity/K
    level_cooccurrence.png          joint usage of level-0 x level-1 codes (+ normalised MI)
    codebook_geometry.png           PCA of latents with the codebook overlaid; code spacing
    code_prototypes.png             level-0 code centroids in feature space; within-code spread
    latent_<pca|tsne|umap>.png      2D projections of encoder latents coloured by features, rank, code
    latent_quantisation_error.png   |z - z_q| and the residual left after each level (+ training reference)
  reconstruction, pre-processed space
    feature_closure.png             input vs reconstructed features, with ratio
    feature_residuals.png           residual distributions, resolution vs the feature value
    feature_correlations.png        feature correlation matrices: input, reconstructed, difference
    resolution_vs_rank.png          res68/std of every feature vs constituent index
  reconstruction, physical space (if the inverse is available)
    feature_closure_physical.png    same as feature_closure, in physical units
    jet_closure.png                 jet observables + per-jet response (4vec, relvec)
    jet_images.png                  mean jet image: input, reconstructed, difference (4vec, relvec)
    lund_plane.png                  Lund-plane density: input, reconstructed, difference (ktdr)
  training (train_vqvae.py only)
    training_history.png            reconstruction, commitment, perplexity, code usage per epoch
  token statistics (--token-stats)
    token_stats.png                 multiplicity, per-position code entropy, code-vs-position map,
                                    usage vs the training reference

Metric conventions
    res68         half-width of the central 68% interval of the residual (robust resolution)
    bias          median residual (median response - 1 for relative/jet quantities)
    w1            1-Wasserstein distance between input and reconstructed marginals / input std
    perplexity    2^H of the code usage per level (= K for perfectly uniform usage)
    level MI      mutual information between level-0 and level-1 codes / min(H0, H1)
    ood_fraction  fraction of constituents whose latent quantisation error exceeds the training
                  99th percentile (~1% in-distribution)
    usage JS      Jensen-Shannon divergence (bits) of code usage vs the training reference
The warning thresholds in FLAG_THRESHOLDS are rules of thumb; tune them to your needs.
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

import helpers_vqvae  # noqa: E402

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

# Feature order per input format (pre-processing tables in helpers.py use the same order)
FORMAT_FEATURES = {
    "4vec": ["E", "p_x", "p_y", "p_z"],
    "relvec": ["p_T frac", "Δη", "Δφ"],
    "ktdr": ["ln k_t", "ln 1/ΔR"],
}
# Role of each feature for the physics view. NOTE: helpers.make_absolute_constituents reads
# relvec as (ptfrac, dphi, deta); pass relvec_order=("ptfrac", "dphi", "deta") if your files do.
RELVEC_ORDER = ("ptfrac", "deta", "dphi")
KTDR_ORDER = ("lnkt", "lndr")

FLAG_THRESHOLDS = dict(
    min_utilisation=0.5,          # fraction of codes used per level
    max_top10_share=0.5,          # share of tokens taken by the 10 most used codes
    max_duplicate_fraction=0.05,  # codes sitting on top of another code
    max_level_mi=0.5,             # normalised MI between level 0 and 1
    max_final_rel_residual=0.3,   # |z - z_q| / |z| after the last level
    min_level_gain=0.05,          # relative residual reduction a level must bring
    max_feature_w1=0.05,          # W1/std for constituent features (both spaces)
    max_corr_diff=0.05,           # largest change of a feature-feature correlation
    max_jet_res68=0.05,           # per-jet response resolution
    max_jet_bias=0.02,            # |median response - 1|
    max_jet_w1=0.05,              # W1/std for jet observables
    max_image_l1=0.15,            # L1 difference of mean jet image / Lund plane
    max_ood_fraction=0.03,        # vs ~0.01 expected in-distribution
    max_usage_js=0.05,            # bits
)


# ---------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------
def _q68(x):
    """median, half-width of central 68% interval"""
    if len(x) == 0:
        return float("nan"), float("nan")
    q16, q50, q84 = np.percentile(x, [16, 50, 84])
    return float(q50), float((q84 - q16) / 2)


def _entropy_bits(counts, miller_madow=False):
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    n = counts.sum()
    p = counts / n
    h = float(-(p * np.log2(p)).sum())
    if miller_madow:  # first-order bias correction for sparse histograms
        h += (counts.size - 1) / (2 * n * np.log(2))
    return h


def _js_bits(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / max(p.sum(), 1e-300), q / max(q.sum(), 1e-300)
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _w1_over_std(t, r, floor=1e-4):
    """W1 / std(input); std floored so (near-)constant features don't give spurious huge values."""
    return float(wasserstein_distance(t, r) / max(float(np.std(t)), floor))


def _wrap_phi(phi):
    return np.remainder(phi + np.pi, 2 * np.pi) - np.pi


def _is_scale_feature(x):
    """Strictly positive and spanning > 2 decades (E, pT fraction, ...): use relative residuals / log axes."""
    if x.size == 0 or x.min() <= 0:
        return False
    lo, hi = np.percentile(x, [0.1, 99.9])
    return hi / max(lo, 1e-300) > 100


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


def default_feature_names(input_format, feature_keys, n_features):
    names = FORMAT_FEATURES.get(input_format)
    if names is not None and len(names) == n_features:
        return list(names)
    if feature_keys is not None and len(feature_keys) == n_features:
        return list(feature_keys)
    return [f"feature {i}" for i in range(n_features)]


# ---------------------------------------------------------------------
# Inverse pre-processing (real constituents only)
# ---------------------------------------------------------------------
def invert_preprocess(f, input_format, method):
    """(M, F) pre-processed features of REAL constituents -> physical features.

    Mirrors helpers.undo_preprocess (same formulas and constant tables) but never re-derives the
    padding from the values: undo_preprocess flags X[..., 0] < 0 as padding for linear/logit,
    which also hits real (reconstructed) values that are slightly negative."""
    f = np.array(f, dtype=np.float64, copy=True)
    if method in (None, "none", "shiftpad"):
        return f
    import helpers  # constant tables; imported lazily (pulls in fastjet)

    def table(fn):
        t = fn(input_format)
        if t is None or len(t) < f.shape[1]:
            raise ValueError(f"helpers.{fn.__name__} has no {f.shape[1]}-feature table for '{input_format}'")
        return t

    if method == "standardize":
        for i, (mu, sd) in enumerate(table(helpers.preprocess_mean_std)[:f.shape[1]]):
            f[:, i] = f[:, i] * sd + mu
    elif method == "linear":
        for i, (a, b) in enumerate(table(helpers.preprocess_min_max)[:f.shape[1]]):
            f[:, i] = (b - a) * f[:, i] + a
    elif method == "logit":
        for i, (a, b) in enumerate(table(helpers.preprocess_min_max)[:f.shape[1]]):
            f[:, i] = (b - a) / (1.0 + np.exp(-f[:, i])) + a
    elif method == "log" and input_format == "relvec":
        f[:, 0] = np.exp(-f[:, 0])
    elif method == "log":
        f = np.sign(f) * np.expm1(np.abs(f))
    elif method == "logstd":
        for i, (mu, sd) in enumerate(table(helpers.preprocess_log_mean_std)[:f.shape[1]]):
            f[:, i] = f[:, i] * sd + mu
        if input_format == "relvec":
            f[:, 0] = np.expm1(f[:, 0])
        else:
            f = np.sign(f) * np.expm1(np.abs(f))
    else:
        raise ValueError(f"unknown pre-processing method '{method}'")
    return f


# ---------------------------------------------------------------------
# Collection: run the tokenizer once and keep everything the metrics/plots need
# ---------------------------------------------------------------------
@torch.no_grad()
def collect(tok, X, mask, n_jets=None, batch_size=4096, device="cpu"):
    """Encode/decode jets and gather per-constituent quantities (numpy, on CPU).

    X: (J, N, F) pre-processed features, mask: (J, N) bool, True = padded."""
    tok.eval()
    dev = torch.device(device)
    X, mask = torch.as_tensor(np.asarray(X)), torch.as_tensor(np.asarray(mask))
    J = len(X) if n_jets is None else min(n_jets, len(X))
    X, mask = X[:J].float(), mask[:J].bool()

    keep = {k: [] for k in ("z", "codes", "rel_res", "qdist", "rec")}
    sq_err, n_seen = torch.zeros(X.shape[-1]), 0
    for i in range(0, J, batch_size):
        x, m = X[i:i + batch_size].to(dev), mask[i:i + batch_size].to(dev)
        real = x[~m]
        if real.shape[0] == 0:
            continue
        z = tok.encoder(real)
        znorm = z.norm(dim=1).clamp_min(EPS)
        residual, cs, rel = z, [], []
        for vq in tok.vq.levels:
            c = vq.quantize(residual)
            residual = residual - vq.lookup(c)
            cs.append(c)
            rel.append(residual.norm(dim=1) / znorm)
        dec = tok.decoder(z - residual)
        sq_err += ((dec - real) ** 2).sum(0).cpu()
        n_seen += real.shape[0]
        keep["rec"].append(dec.cpu())
        keep["z"].append(z.cpu())
        keep["codes"].append(torch.stack(cs, -1).cpu())
        keep["rel_res"].append(torch.stack(rel, -1).cpu())
        keep["qdist"].append(residual.norm(dim=1).cpu())

    real = ~mask
    N = mask.shape[1]
    return {
        "K": tok.K, "Q": tok.Q, "N": N, "F": X.shape[-1], "n_jets": J,
        "z": torch.cat(keep["z"]).numpy(), "codes": torch.cat(keep["codes"]).numpy(),
        "rel_res": torch.cat(keep["rel_res"]).numpy(), "qdist": torch.cat(keep["qdist"]).numpy(),
        "f_true": X[real].numpy().astype(np.float64), "f_rec": torch.cat(keep["rec"]).numpy().astype(np.float64),
        "rank": torch.arange(N).expand(J, N)[real].numpy(),
        "jet_idx": torch.arange(J).unsqueeze(1).expand(J, N)[real].numpy(),
        "mult": real.sum(1).numpy(),
        "mse": (sq_err / max(n_seen, 1)).numpy(),
        "codebooks": [vq.embed.detach().cpu().numpy() for vq in tok.vq.levels],
    }

# ---------------------------------------------------------------------
# Physics view
# ---------------------------------------------------------------------
def _jet_sums(p4, jet_idx, J):
    S = np.stack([np.bincount(jet_idx, weights=p4[:, k], minlength=J) for k in range(4)], -1)
    E, px, py, pz = S.T
    pt = np.hypot(px, py)
    return {"pt": pt, "eta": np.arcsinh(pz / np.maximum(pt, EPS)), "phi": np.arctan2(py, px),
            "mass": np.sqrt(np.clip(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0, None))}


def _coords(p4, jet_idx, axis):
    """per-constituent (d eta, d phi, pt) w.r.t. a given jet axis"""
    cpt = np.hypot(p4[:, 1], p4[:, 2])
    ceta = np.arcsinh(p4[:, 3] / np.maximum(cpt, EPS))
    cphi = np.arctan2(p4[:, 2], p4[:, 1])
    return ceta - axis["eta"][jet_idx], _wrap_phi(cphi - axis["phi"][jet_idx]), cpt


def _jet_observables(p4, jet_idx, J):
    """jet pT, mass, girth, pT_D from (M, 4) real constituents, each jet w.r.t. its own axis"""
    s = _jet_sums(p4, jet_idx, J)
    deta, dphi, cpt = _coords(p4, jet_idx, s)
    dr = np.hypot(deta, dphi)
    sum_pt = np.bincount(jet_idx, weights=cpt, minlength=J)
    girth = np.bincount(jet_idx, weights=cpt * dr, minlength=J) / np.maximum(s["pt"], EPS)
    ptd = np.sqrt(np.bincount(jet_idx, weights=cpt ** 2, minlength=J)) / np.maximum(sum_pt, EPS)
    return {"pt": s["pt"], "mass": s["mass"], "girth": girth, "ptd": ptd}, s


def _massless_p4(pt, eta, phi):
    return np.stack([pt * np.cosh(eta), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], -1)


def _image(x, y, w, J, edges_x, edges_y):
    img, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y], weights=w)
    return img / max(J, 1)


def physics_view(col, input_format, preprocess, relvec_order=RELVEC_ORDER, ktdr_order=KTDR_ORDER):
    """Physical-space features and per-format physics. Returns None (with a printed reason) if the
    inverse pre-processing is not available."""
    try:
        p_true = invert_preprocess(col["f_true"], input_format, preprocess)
        p_rec = invert_preprocess(col["f_rec"], input_format, preprocess)
    except (ImportError, ValueError) as e:
        print(f"  physical-space diagnostics skipped: {e}")
        return None
    phys = {"p_true": p_true, "p_rec": p_rec, "kind": None}
    J, jidx = col["n_jets"], col["jet_idx"]
    finite = np.isfinite(p_true).all(1) & np.isfinite(p_rec).all(1)
    if not finite.all():
        print(f"  note: {int((~finite).sum())} constituents non-finite after inverse pre-processing")

    if input_format == "4vec" and col["F"] == 4:
        obs_t, axis = _jet_observables(p_true, jidx, J)
        obs_r, _ = _jet_observables(p_rec, jidx, J)
        deta_t, dphi_t, pt_t = _coords(p_true, jidx, axis)  # both w.r.t. the INPUT jet axis
        deta_r, dphi_r, pt_r = _coords(p_rec, jidx, axis)
        w_t, w_r = pt_t / np.maximum(axis["pt"][jidx], EPS), pt_r / np.maximum(axis["pt"][jidx], EPS)
        phys.update(kind="jets", jets_true=obs_t, jets_rec=obs_r,
                    jet_labels={"pt": r"jet $p_T$", "mass": "jet mass", "girth": "girth", "ptd": r"$p_T^D$"})
        for tag, p4 in (("true", p_true), ("rec", p_rec)):  # mass shell: E, px, py, pz are quantised independently
            m2 = p4[:, 0] ** 2 - (p4[:, 1:] ** 2).sum(1)
            phys[f"m_over_E_{tag}"] = np.sqrt(np.abs(m2)) / np.maximum(np.abs(p4[:, 0]), EPS)
            phys[f"m2_negative_{tag}"] = float((m2 < 0).mean())
    elif input_format == "relvec" and col["F"] == 3:
        i = {k: relvec_order.index(k) for k in ("ptfrac", "deta", "dphi")}
        p4_t = _massless_p4(p_true[:, i["ptfrac"]], p_true[:, i["deta"]], p_true[:, i["dphi"]])
        p4_r = _massless_p4(p_rec[:, i["ptfrac"]], p_rec[:, i["deta"]], p_rec[:, i["dphi"]])
        obs_t, _ = _jet_observables(p4_t, jidx, J)  # jet frame: pT_J = 1, mass = m/pT_J (boost invariant)
        obs_r, _ = _jet_observables(p4_r, jidx, J)
        deta_t, dphi_t, w_t = p_true[:, i["deta"]], p_true[:, i["dphi"]], p_true[:, i["ptfrac"]]
        deta_r, dphi_r, w_r = p_rec[:, i["deta"]], p_rec[:, i["dphi"]], p_rec[:, i["ptfrac"]]
        phys.update(kind="jets", jets_true=obs_t, jets_rec=obs_r,
                    jet_labels={"pt": r"$|\sum \vec p_T| / p_{T,J}$", "mass": r"$m / p_{T,J}$",
                                "girth": "girth", "ptd": r"$p_T^D$"})
    elif input_format == "ktdr" and col["F"] == 2:
        i = {k: ktdr_order.index(k) for k in ("lnkt", "lndr")}
        x_t, y_t = p_true[:, i["lndr"]], p_true[:, i["lnkt"]]
        x_r, y_r = p_rec[:, i["lndr"]], p_rec[:, i["lnkt"]]
        ex = np.linspace(*np.percentile(x_t[finite], [0.2, 99.8]), 31)
        ey = np.linspace(*np.percentile(y_t[finite], [0.2, 99.8]), 31)
        phys.update(kind="lund", img_true=_image(x_t, y_t, None, J, ex, ey), img_rec=_image(x_r, y_r, None, J, ex, ey),
                    img_extent=[ex[0], ex[-1], ey[0], ey[-1]], img_labels=(r"$\ln(1/\Delta R)$", r"$\ln k_t$"),
                    img_title="Lund-plane density (emissions per jet per bin)")
        return phys
    else:
        return phys

    # jet image for 4vec / relvec
    ok = finite
    R = float(max(0.4, np.percentile(np.hypot(deta_t[ok], dphi_t[ok]), 99.5)))
    e = np.linspace(-R, R, 26)  # coarse: finer pixels mostly show bin-migration noise
    phys.update(img_true=_image(deta_t[ok], dphi_t[ok], w_t[ok], J, e, e),
                img_rec=_image(deta_r[ok], dphi_r[ok], w_r[ok], J, e, e),
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
        ok = np.isfinite(tj) & np.isfinite(rj)
        tj, rj = tj[ok], rj[ok]
        if relative_scale and _is_scale_feature(tj):
            med, r68 = _q68(rj / tj)
            out[name] = {"type": "relative", "bias": med - 1, "res68": r68, "w1": _w1_over_std(tj, rj)}
        else:
            res = rj - tj
            if name in ("Δφ", "dphi", "DPHI"):
                res = _wrap_phi(res)
            b, r68 = _q68(res)
            sd = max(float(np.std(tj)), 1e-12)
            out[name] = {"type": "absolute", "bias": b, "res68": r68, "res68_over_std": r68 / sd,
                         "rms": float(np.sqrt((res ** 2).mean())), "w1": _w1_over_std(tj, rj)}
    return out


def compute_metrics(col, names, phys=None, reference=None, token_stats=False):
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
        E = col["codebooks"][q][used]
        if len(E) > 1:
            d = torch.cdist(torch.from_numpy(E), torch.from_numpy(E))
            d.fill_diagonal_(float("inf"))
            nn = d.min(1).values.numpy()
            dup = int((nn < 1e-2 * (float(np.median(np.linalg.norm(E, axis=1))) + 1e-12)).sum())
        else:
            nn, dup = np.array([np.nan]), 0
        lvl = {"used": int(used.sum()), "utilisation": float(used.mean()), "dead": int((~used).sum()),
               "entropy_bits": H, "perplexity": float(2 ** H), "perplexity_over_K": float(2 ** H / K),
               "top10_share": float(np.sort(counts)[::-1][:10].sum() / counts.sum()),
               "rel_residual_after_mean": float(col["rel_res"][:, q].mean()),
               "nn_distance_median": float(np.nanmedian(nn)), "duplicate_codes": dup}
        if reference is not None and q < len(reference.get("usage", [])):
            lvl["usage_js_vs_train"] = _js_bits(counts, reference["usage"][q])
        levels.append(lvl)
    m["levels"] = levels
    m["bits_per_constituent"] = float(sum(Hs))
    m["max_bits_per_constituent"] = float(Q * np.log2(K))
    if Q >= 2:
        H0 = _entropy_bits(np.bincount(codes[:, 0], minlength=K), True)
        H1 = _entropy_bits(np.bincount(codes[:, 1], minlength=K), True)
        mi = max(0.0, H0 + H1 - _entropy_bits(np.bincount(codes[:, 0] * K + codes[:, 1], minlength=K * K), True))
        m["level01_mi_bits"], m["level01_mi_normalised"] = mi, float(mi / max(min(H0, H1), 1e-12))

    qd = col["qdist"]
    m["latent_qdist_median"], m["latent_qdist_q99"] = float(np.median(qd)), float(np.percentile(qd, 99))
    if reference is not None and "qdist_q99" in reference:
        m["ood_fraction"] = float((qd > reference["qdist_q99"]).mean())

    # reconstruction, pre-processed space
    m["recon_mse"] = float(col["mse"].mean())  # same quantity as the training loss
    m["preprocessed"] = _feature_metrics(col["f_true"], col["f_rec"], names)
    for j, name in enumerate(names):
        m["preprocessed"][name]["mse"] = float(col["mse"][j])
    ct, cr = np.corrcoef(col["f_true"].T), np.corrcoef(col["f_rec"].T)
    m["corr_true"], m["corr_rec"] = np.atleast_2d(ct), np.atleast_2d(cr)
    m["corr_max_abs_diff"] = float(np.nanmax(np.abs(np.atleast_2d(cr - ct)))) if col["F"] > 1 else 0.0
    res = col["f_rec"] - col["f_true"]
    sd = np.maximum(col["f_true"].std(0), 1e-12)
    m["res68_over_std_vs_rank"] = {
        name: [_q68(res[col["rank"] == r, j])[1] / sd[j] for r in range(col["N"])] for j, name in enumerate(names)}

    # physical space
    if phys is not None:
        m["physical"] = _feature_metrics(phys["p_true"], phys["p_rec"], names, relative_scale=True)
        if phys["kind"] == "jets":
            jets = {}
            for key in phys["jets_true"]:
                t, r = phys["jets_true"][key], phys["jets_rec"][key]
                ok = (t > 1e-3 * max(np.median(t), 1e-12)) & np.isfinite(r)
                med, r68 = _q68(r[ok] / t[ok])
                const = np.std(t[ok]) < 0.01 * abs(np.median(t[ok]))  # e.g. |sum pT|/pT_J = 1 by construction
                jets[key] = {"response_median": med, "response_res68": r68,
                             "w1": None if const else _w1_over_std(t[ok], r[ok]), "input_near_constant": bool(const)}
            m["jet"] = jets
            if "m_over_E_true" in phys:
                m["constituent_mass"] = {
                    "median_m_over_E_input": float(np.median(phys["m_over_E_true"])),
                    "median_m_over_E_reco": float(np.median(phys["m_over_E_rec"])),
                    "frac_m2_negative_input": phys["m2_negative_true"], "frac_m2_negative_reco": phys["m2_negative_rec"]}
        if "img_true" in phys:
            it, ir = phys["img_true"], phys["img_rec"]
            m["image_l1"] = float(np.abs(ir - it).sum() / max(it.sum(), 1e-12))

    if token_stats:
        m["tokens"] = token_metrics(col)
    m["flags"] = flags(m)
    return m


def token_metrics(col):
    """Sequence statistics relevant for the next-token model (level-0 codes)."""
    K, N = col["K"], col["N"]
    c0, rank, jet = col["codes"][:, 0], col["rank"], col["jet_idx"]
    pos_H = [_entropy_bits(np.bincount(c0[rank == r], minlength=K), True) if (rank == r).sum() > 1 else float("nan")
             for r in range(N)]
    same = jet[1:] == jet[:-1]
    prev, nxt = c0[:-1][same], c0[1:][same]
    H_next = _entropy_bits(np.bincount(nxt, minlength=K), True)
    H_joint = _entropy_bits(np.bincount(prev * K + nxt, minlength=K * K), True)
    H_prev = _entropy_bits(np.bincount(prev, minlength=K), True)
    mult = col["mult"]
    return {"tokens_per_jet_mean": float(mult.mean() * col["Q"]), "tokens_per_jet_max": int(mult.max() * col["Q"]),
            "multiplicity_mean": float(mult.mean()), "multiplicity_std": float(mult.std()),
            "level0_entropy_per_position_bits": pos_H,
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
        if lvl["used"] and lvl["duplicate_codes"] / lvl["used"] > T["max_duplicate_fraction"]:
            out.append(f"level {q}: {lvl['duplicate_codes']} near-duplicate codes -> wasted capacity")
        if q > 0:
            prev = m["levels"][q - 1]["rel_residual_after_mean"]
            gain = 1 - lvl["rel_residual_after_mean"] / max(prev, 1e-12)
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
        for key, v in m.get(space, {}).items():
            if v["w1"] > T["max_feature_w1"]:
                out.append(f"{space} {key}: reconstructed distribution off (W1/std {v['w1']:.3f})")
    if m["corr_max_abs_diff"] > T["max_corr_diff"]:
        out.append(f"feature correlations not preserved (max |Δρ| = {m['corr_max_abs_diff']:.3f})")
    for key, v in m.get("jet", {}).items():
        if v["response_res68"] > T["max_jet_res68"]:
            out.append(f"jet {key}: response resolution {v['response_res68']:.1%}")
        if abs(v["response_median"] - 1) > T["max_jet_bias"]:
            out.append(f"jet {key}: response biased ({v['response_median'] - 1:+.1%})")
        if v["w1"] is not None and v["w1"] > T["max_jet_w1"]:
            out.append(f"jet {key}: reconstructed distribution off (W1/std {v['w1']:.3f})")
    cm = m.get("constituent_mass")
    if cm and cm["median_m_over_E_reco"] > max(3 * cm["median_m_over_E_input"], 0.02):
        out.append(f"reconstructed constituents are off mass shell (median m/E {cm['median_m_over_E_reco']:.3f} vs "
                   f"{cm['median_m_over_E_input']:.3f} input, {cm['frac_m2_negative_reco']:.0%} with m^2 < 0): E and p "
                   f"are quantised independently -> biases jet mass; prefer relvec, or project E onto |p| after decoding")
    if m.get("image_l1", 0) > T["max_image_l1"]:
        out.append(f"mean jet image / Lund plane differs (L1 {m['image_l1']:.3f})")
    if m.get("ood_fraction", 0) > T["max_ood_fraction"]:
        out.append(f"{m['ood_fraction']:.1%} of constituents beyond the training 99th-percentile quantisation "
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
    print(f"  recon MSE {m['recon_mse']:.5f}; max |Δ correlation| {m['corr_max_abs_diff']:.4f}")
    print("  pre-processed (res68/std, W1/std): " + ", ".join(
        f"{k} {v['res68_over_std']:.4f}/{v['w1']:.3f}" for k, v in m["preprocessed"].items()))
    if "physical" in m:
        print("  physical (bias, res68, W1/std): " + ", ".join(
            f"{k} {v['bias']:+.4f} {v['res68']:.4f}{' rel' if v['type'] == 'relative' else ''} {v['w1']:.3f}"
            for k, v in m["physical"].items()))
    if "jet" in m:
        print("  jets (response median ± res68, W1/std): " + ", ".join(
            f"{k} {v['response_median']:.3f}±{v['response_res68']:.3f} "
            f"({'const' if v['w1'] is None else format(v['w1'], '.3f')})" for k, v in m["jet"].items()))
    if "constituent_mass" in m:
        cm = m["constituent_mass"]
        print(f"  constituent m/E median: input {cm['median_m_over_E_input']:.4f}, reco {cm['median_m_over_E_reco']:.4f}"
              f" (m^2 < 0: {cm['frac_m2_negative_reco']:.1%})")
    if "image_l1" in m:
        print(f"  image L1 difference {m['image_l1']:.3f}")
    if "ood_fraction" in m:
        print(f"  OOD: {m['ood_fraction']:.2%} of constituents beyond training q99 quantisation error (~1% expected)")
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
    x = x[np.isfinite(x)]
    if log:
        lo, hi = np.percentile(x, [0.1, 99.9])
        return np.logspace(np.log10(lo) - 0.05, np.log10(hi) + 0.05, n + 1)
    lo, hi = np.percentile(x, [0.1, 99.9])
    pad = 0.05 * (hi - lo + 1e-12)
    return np.linspace(lo - pad, hi + pad, n + 1)


def _constant(x):
    return np.ptp(x) < 1e-5


def _placeholder(ax, title, text):
    ax.set_title(title)
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=INK2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


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


def _grid(n, max_cols=4):
    ncols = min(n, max_cols)
    return ncols, int(np.ceil(n / ncols))


def plot_feature_closure(t, r, names, metrics, outdir, fname, title, auto_log=False):
    F = len(names)
    ncols, nrows = _grid(F)
    fig = plt.figure(figsize=(3.8 * ncols, 4.4 * nrows))
    gs = fig.add_gridspec(2 * nrows, ncols, height_ratios=[3, 1] * nrows, hspace=0.08, wspace=0.28)
    handles = None
    for j, name in enumerate(names):
        row, colm = divmod(j, ncols)
        ax = fig.add_subplot(gs[2 * row, colm])
        axr = fig.add_subplot(gs[2 * row + 1, colm], sharex=ax)
        tj, rj = t[:, j], r[:, j]
        ok = np.isfinite(tj) & np.isfinite(rj)
        tj, rj = tj[ok], rj[ok]
        if _constant(tj) and _constant(rj):
            _placeholder(ax, name, "constant")
            axr.axis("off")
            continue
        log = auto_log and _is_scale_feature(tj) and rj.min() > 0
        bins = _bins(tj, log=log)
        ht, hr = _pair_hist(ax, tj, rj, bins)
        _ratio(axr, bins, ht, hr, log)
        if log:
            ax.set_xscale("log")
        ax.set_title(f"{name}   W1/σ = {metrics[name]['w1']:.3f}")
        ax.tick_params(labelbottom=False)
        axr.set_xlabel(name)
        if colm == 0:
            ax.set_ylabel("constituents")
            axr.set_ylabel("reco / input")
        handles = handles or ax.get_legend_handles_labels()
    if handles:
        fig.legend(*handles, loc="upper right", ncol=2, bbox_to_anchor=(0.91, 1.0))
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
        if _constant(r):
            _placeholder(ax, f"residual {name}", "exactly 0")
            ax2.axis("off")
            continue
        lim = np.percentile(np.abs(r), 99.5) * 1.1 + 1e-12
        ax.hist(r, np.linspace(-lim, lim, 81), histtype="stepfilled", color=TRUE_C, alpha=0.85)
        ax.set_yscale("log")
        ax.axvline(0, color=AXIS, lw=1)
        v = m["preprocessed"][name]
        ax.set_title(f"reco − input: {name}")
        ax.text(0.03, 0.95, f"bias {v['bias']:+.4f}\nres68 {v['res68']:.4f}", transform=ax.transAxes,
                va="top", color=INK2, fontsize=8)
        edges = np.unique(np.percentile(t, np.linspace(0, 100, 13)))
        if len(edges) > 2:
            which = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)
            prof = [_q68(r[which == b])[1] for b in range(len(edges) - 1)]
            ax2.plot(0.5 * (edges[1:] + edges[:-1]), prof, "-o", color=TRUE_C, ms=5)
        ax2.set_xlabel(f"input {name}")
        ax2.set_title(f"resolution of {name} vs its value")
        ax2.set_ylim(bottom=0)
        if j == 0:
            ax.set_ylabel("constituents")
            ax2.set_ylabel("res68")
    fig.suptitle("Residuals in pre-processed space (the space the loss is computed in)", x=0.08, ha="left", y=0.99)
    return _save(fig, outdir, "feature_residuals.png")


def plot_feature_correlations(m, names, outdir):
    ct, cr = np.atleast_2d(m["corr_true"]), np.atleast_2d(m["corr_rec"])
    d = cr - ct
    v = max(np.nanmax(np.abs(d)), 1e-3)
    F = len(names)
    fig, axes = plt.subplots(1, 3, figsize=(4.4 * 3 + 1.5, 4.2), gridspec_kw={"wspace": 0.45})
    panels = ((ct, "input", DIV, TwoSlopeNorm(0, -1, 1)), (cr, "reconstructed", DIV, TwoSlopeNorm(0, -1, 1)),
              (d, f"reco − input (max |Δρ| = {m['corr_max_abs_diff']:.3f})", DIV, TwoSlopeNorm(0, -v, v)))
    for ax, (mat, title, cmap, norm) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, norm=norm)
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
    r = np.arange(1, len(next(iter(m["res68_over_std_vs_rank"].values()))) + 1)
    for j, name in enumerate(names):
        ax.plot(r, m["res68_over_std_vs_rank"][name], "-o", ms=5, color=SERIES[j % 8], label=name)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("constituent index (stored order)")
    ax.set_ylabel("res68 / std of feature")
    ax.set_title("Reconstruction resolution vs constituent index (pre-processed space)")
    ax.legend()
    return _save(fig, outdir, "resolution_vs_rank.png")


def plot_jet_closure(phys, m, outdir):
    keys = list(phys["jets_true"])
    labels = phys["jet_labels"]
    fig, axes = plt.subplots(2, len(keys), figsize=(3.8 * len(keys), 6.4), gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    for j, key in enumerate(keys):
        t, r = phys["jets_true"][key], phys["jets_rec"][key]
        ok = np.isfinite(t) & np.isfinite(r)
        ax, ax2 = axes[0, j], axes[1, j]
        _pair_hist(ax, t[ok], r[ok], _bins(t[ok], 50))
        v = m["jet"][key]
        ax.set_title(labels[key] + ("" if v["w1"] is None else f"   W1/σ = {v['w1']:.3f}"))
        ax.set_xlabel(labels[key])
        ok &= t > 1e-3 * max(np.median(t), 1e-12)
        resp = r[ok] / t[ok]
        half = max(4 * v["response_res68"], 0.01)
        ax2.hist(resp, np.linspace(v["response_median"] - half, v["response_median"] + half, 61),
                 histtype="stepfilled", color=TRUE_C, alpha=0.85)
        ax2.axvline(1, color=INK2, lw=1, ls="--")
        ax2.set_title(f"response {labels[key]}")
        ax2.set_xlabel("reco / input (per jet)")
        ax2.text(0.03, 0.95, f"median {v['response_median']:.4f}\nres68 {v['response_res68']:.4f}",
                 transform=ax2.transAxes, va="top", color=INK2, fontsize=8)
        if j == 0:
            ax.set_ylabel("jets")
            ax2.set_ylabel("jets")
            fig.legend(*ax.get_legend_handles_labels(), loc="upper right", ncol=2, bbox_to_anchor=(0.91, 0.99))
    return _save(fig, outdir, "jet_closure.png")


def plot_density_images(phys, m, outdir, fname):
    it, ir, ext = phys["img_true"], phys["img_rec"], phys["img_extent"]
    xl, yl = phys["img_labels"]
    pos = it[it > 0]
    vmin = max(pos.min() if pos.size else 1e-8, it.max() * 1e-4)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"wspace": 0.35})
    for ax, img, title in ((axes[0], it, "input"), (axes[1], ir, "reconstructed")):
        im = ax.imshow(np.clip(img.T, vmin, None), origin="lower", extent=ext, cmap=SEQ, aspect="auto",
                       norm=LogNorm(vmin=vmin, vmax=it.max()))
        ax.set_title(f"{phys['img_title']}: {title}", fontsize=9)
        fig.colorbar(im, ax=ax, label="per jet per bin")
    d = (ir - it) / max(it.max(), 1e-12)
    v = max(np.abs(d).max(), 1e-6)
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


def plot_codebook_geometry(col, outdir, max_points=20000):
    from sklearn.decomposition import PCA
    K, Q = col["K"], col["Q"]
    idx = _subsample(len(col["z"]), max_points)
    pca = PCA(2).fit(col["z"][idx])
    pz = pca.transform(col["z"][idx])
    counts = np.bincount(col["codes"][:, 0], minlength=K)
    pc = pca.transform(col["codebooks"][0])
    used = counts > 0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.3})
    a1.scatter(pz[:, 0], pz[:, 1], s=4, color=MUTED, alpha=0.35, lw=0, label="encoder latents")
    a1.scatter(pc[used, 0], pc[used, 1], s=4 + 50 * np.sqrt(counts[used] / counts.max()), color=SERIES[0],
               edgecolor=SURFACE, lw=0.5, alpha=0.9, label="level-0 codes (size ∝ usage)")
    if (~used).any():
        a1.scatter(pc[~used, 0], pc[~used, 1], s=20, marker="x", color=CRITICAL, lw=1.2, label="unused codes")
    ev = pca.explained_variance_ratio_
    a1.set_xlabel(f"PC1 ({ev[0]:.0%} var.)")
    a1.set_ylabel(f"PC2 ({ev[1]:.0%} var.)")
    a1.set_title("Latent space (PCA) with level-0 codebook")
    a1.legend(loc="best")
    for q in range(Q):
        E = col["codebooks"][q][np.bincount(col["codes"][:, q], minlength=K) > 0]
        if len(E) < 2:
            continue
        d = torch.cdist(torch.from_numpy(E), torch.from_numpy(E))
        d.fill_diagonal_(float("inf"))
        nn = d.min(1).values.numpy()
        nn = nn[nn > 0]
        if nn.size:
            a2.hist(nn, np.logspace(np.log10(nn.min()), np.log10(nn.max()) + 1e-9, 50), histtype="step",
                    lw=2, color=SERIES[q % 8], label=f"level {q}")
    a2.set_xscale("log")
    a2.set_xlabel("distance to nearest other code")
    a2.set_ylabel("codes")
    a2.set_title("Codebook spacing (spike at small distance = duplicates)")
    a2.legend()
    return _save(fig, outdir, "codebook_geometry.png")


def _prototype_axes(input_format, F, relvec_order=RELVEC_ORDER):
    """(x, y, colour) feature indices for the code-centroid scatter."""
    if input_format == "4vec" and F == 4:
        return 1, 2, 0
    if input_format == "relvec" and F == 3:
        return relvec_order.index("deta"), relvec_order.index("dphi"), relvec_order.index("ptfrac")
    if input_format == "ktdr" and F == 2:
        return KTDR_ORDER.index("lndr"), KTDR_ORDER.index("lnkt"), None
    return 0, min(1, F - 1), (2 if F > 2 else None)


def plot_code_prototypes(col, names, input_format, outdir, relvec_order=RELVEC_ORDER):
    K, F = col["K"], col["F"]
    c0, f = col["codes"][:, 0], col["f_true"]
    n = np.bincount(c0, minlength=K).astype(float)
    used = n > 0
    mean = np.stack([np.bincount(c0, weights=f[:, j], minlength=K)[used] / n[used] for j in range(F)], -1)
    var = np.stack([np.bincount(c0, weights=f[:, j] ** 2, minlength=K)[used] / n[used] for j in range(F)], -1) - mean ** 2
    spread = np.sqrt(np.clip(var, 0, None)) / np.maximum(f.std(0), 1e-12)
    size = 4 + 60 * np.sqrt(n[used] / n.max())
    xi, yi, ci = _prototype_axes(input_format, F, relvec_order)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.3})
    if F == 1:
        _placeholder(a1, "Level-0 code centroids", "single feature")
    elif ci is None:
        a1.scatter(mean[:, xi], mean[:, yi], s=size, color=SERIES[0], edgecolor=SURFACE, lw=0.5, alpha=0.85)
    else:
        sc = a1.scatter(mean[:, xi], mean[:, yi], c=mean[:, ci], s=size, cmap=SEQ, edgecolor=SURFACE, lw=0.5)
        fig.colorbar(sc, ax=a1, label=f"mean {names[ci]} (pre-processed)")
    if F > 1:
        a1.set_xlabel(f"mean {names[xi]} (pre-processed)")
        a1.set_ylabel(f"mean {names[yi]} (pre-processed)")
        a1.set_title("Level-0 code centroids in feature space (size ∝ usage)")
    hi = np.percentile(spread, 99.5) * 1.05 + 1e-9
    for j, name in enumerate(names):
        a2.hist(spread[:, j], np.linspace(0, hi, 50), histtype="step", lw=2, color=SERIES[j % 8], label=name)
    a2.set_xlabel("within-code std / feature std")
    a2.set_ylabel("level-0 codes")
    a2.set_title("How wide each code is, level 0 only (later levels refine)")
    a2.legend()
    return _save(fig, outdir, "code_prototypes.png")


def _project(X, method):
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(2).fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(2, perplexity=min(30.0, (len(X) - 1) / 3), init="pca", learning_rate="auto",
                    random_state=0).fit_transform(X)
    if method == "umap":
        import umap  # optional: pip install umap-learn
        return umap.UMAP(n_components=2, random_state=0).fit_transform(X)
    raise ValueError(method)


def plot_latent_projection(col, names, outdir, method, max_points=5000):
    idx = _subsample(len(col["z"]), max_points, seed=1)
    try:
        P = _project(col["z"][idx], method)
    except ImportError:
        print(f"  skipping {method}: package not installed (pip install umap-learn)")
        return None
    f = col["f_true"][idx]
    colourings = [(f[:, j], f"{names[j]} (pre-processed)") for j in range(min(col["F"], 3))]
    colourings.append((col["rank"][idx] + 1, "constituent index"))
    n = len(colourings) + 1
    fig, axes = plt.subplots(1, n, figsize=(4.7 * n, 4.4), gridspec_kw={"wspace": 0.25})
    for ax, (c, label) in zip(axes, colourings):
        lo, hi = np.percentile(c, [1, 99])
        sc = ax.scatter(P[:, 0], P[:, 1], c=c, s=3, cmap=SEQ, vmin=lo, vmax=hi, lw=0)
        fig.colorbar(sc, ax=ax, label=label)
        ax.set_title(f"coloured by {label}")
    c0 = col["codes"][idx, 0]
    top = np.argsort(-np.bincount(c0))[:8]
    other = ~np.isin(c0, top)
    ax = axes[-1]
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
    fig.suptitle(f"Encoder latents, {method.upper()} projection ({len(idx)} constituents). Smooth gradients = "
                 "organised latent space; codes should tile it without large gaps", x=0.06, ha="left", y=1.02)
    return _save(fig, outdir, f"latent_{method}.png")


def plot_quantisation_error(col, m, outdir, reference=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"wspace": 0.3})
    qd = col["qdist"][col["qdist"] > 0]
    a1.hist(qd, np.logspace(np.log10(np.percentile(qd, 0.1)), np.log10(qd.max()), 70), histtype="stepfilled",
            color=TRUE_C, alpha=0.85, label="this data")
    if reference is not None and "qdist_q99" in reference:
        a1.axvline(reference["qdist_q99"], color=INK2, ls="--", lw=1.2, label="training q99")
        a1.text(0.97, 0.95, f"beyond training q99: {m['ood_fraction']:.2%}\n(~1% expected)",
                transform=a1.transAxes, ha="right", va="top", color=INK2, fontsize=8)
    a1.set_xscale("log")
    a1.set_xlabel(r"latent quantisation error $|z - z_q|$")
    a1.set_ylabel("constituents")
    a1.set_title("Quantisation error (long tail = poorly covered region / OOD)")
    a1.legend(loc="upper left")
    rr = col["rel_res"]
    lo = max(np.percentile(rr[rr > 0], 0.1), 1e-4) if (rr > 0).any() else 1e-4
    bins = np.logspace(np.log10(lo), np.log10(max(rr.max(), lo * 10)), 70)
    for q in range(col["Q"]):
        a2.hist(rr[:, q], bins, histtype="step", lw=2, color=SERIES[q % 8],
                label=f"after level {q} (mean {m['levels'][q]['rel_residual_after_mean']:.3f})")
    a2.set_xscale("log")
    a2.set_xlabel(r"remaining residual $|z - \sum_{\leq q} e_q| / |z|$")
    a2.set_title("What each RVQ level removes (curves should step left)")
    a2.legend()
    return _save(fig, outdir, "latent_quantisation_error.png")


def _hist_get(h, *keys):
    for k in keys:
        if k in h:
            return h[k]
    return np.nan


def plot_training_history(history, outdir):
    ep = [h["epoch"] for h in history]
    Q, K = len(history[0]["perplexity"]), history[0]["K"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw={"hspace": 0.4, "wspace": 0.28})
    for ax, what, title in ((axes[0, 0], "reco", "Reconstruction MSE"), (axes[0, 1], "commit", "Commitment loss")):
        ax.plot(ep, [_hist_get(h, f"train_{what}", f"train_{what}n") for h in history], color=SERIES[0], label="train")
        ax.plot(ep, [_hist_get(h, f"test_{what}", f"val_{what}", f"val_{what}n") for h in history], color=SERIES[1],
                label="test")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend()
    for q in range(Q):
        axes[1, 0].plot(ep, [h["perplexity"][q] for h in history], color=SERIES[q % 8], label=f"level {q}")
        axes[1, 1].plot(ep, [h["used"][q] / K for h in history], color=SERIES[q % 8], label=f"level {q}")
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
    for q in range(col["Q"]):
        cq = col["codes"][:, q]
        Hq = [_entropy_bits(np.bincount(cq[col["rank"] == r], minlength=K), True) if (col["rank"] == r).sum() > 1
              else np.nan for r in range(N)]
        a.plot(np.arange(1, N + 1), Hq, "-o", ms=5, color=SERIES[q % 8], label=f"level {q}")
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
    for r in range(N):
        sel = col["rank"] == r
        if sel.any():
            H[:, r] = np.bincount(c0[sel], minlength=K)[top] / sel.sum()
    im = a.imshow(H, aspect="auto", origin="lower", cmap=SEQ, extent=[0.5, N + 0.5, -0.5, len(top) - 0.5],
                  interpolation="nearest")
    fig.colorbar(im, ax=a, label="fraction of constituents at this position")
    a.set_xlabel("constituent index")
    a.set_ylabel("level-0 code (top 64 by usage)")
    a.set_title("Code vs position")
    a.grid(False)
    a = axes[1, 1]
    if reference is None or "usage" not in reference:
        _placeholder(a, "Usage vs training", "no training reference\nin this tokenizer checkpoint")
    else:
        for q in range(col["Q"]):
            u = np.bincount(col["codes"][:, q], minlength=K) / len(c0)
            ref = np.asarray(reference["usage"][q])
            ok = (u > 0) & (ref > 0)
            a.scatter(ref[ok], u[ok], s=10, color=SERIES[q % 8], lw=0, alpha=0.8,
                      label=f"level {q} (JS {m['levels'][q]['usage_js_vs_train']:.4f} bits)")
        lo = 0.5 / max(len(c0), 1)
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
def run_diagnostics(col, outdir, input_format=None, preprocess=None, feature_names=None, feature_keys=None,
                    reference=None, history=None, token_stats=False, plots=True, physical=True,
                    projections=("pca", "tsne"), max_points=5000, relvec_order=RELVEC_ORDER, verbose=True):
    """Compute metrics (always) and plots (if plots=True) from a collect() result.
    Returns the metrics dict; writes <outdir>/metrics.json."""
    t0 = time.time()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = feature_names or default_feature_names(input_format, feature_keys, col["F"])
    phys = physics_view(col, input_format, preprocess, relvec_order) if physical else None
    m = compute_metrics(col, names, phys, reference, token_stats=token_stats)
    m["input_format"], m["preprocess"] = input_format, preprocess
    if plots:
        with plt.rc_context(RC):
            paths = [
                plot_feature_closure(col["f_true"], col["f_rec"], names, m["preprocessed"], outdir,
                                     "feature_closure.png", "Constituent features, pre-processed space"),
                plot_feature_residuals(col, m, names, outdir), plot_resolution_vs_rank(m, names, outdir),
                plot_codebook_usage(col, m, outdir), plot_codebook_geometry(col, outdir),
                plot_code_prototypes(col, names, input_format, outdir, relvec_order),
                plot_quantisation_error(col, m, outdir, reference)]
            if col["F"] > 1:
                paths.append(plot_feature_correlations(m, names, outdir))
            if col["Q"] >= 2:
                paths.append(plot_level_cooccurrence(col, m, outdir))
            if phys is not None:
                paths.append(plot_feature_closure(phys["p_true"], phys["p_rec"], names, m["physical"], outdir,
                                                  "feature_closure_physical.png",
                                                  f"Constituent features, physical space (inverse of '{preprocess}')",
                                                  auto_log=True))
                if phys["kind"] == "jets":
                    paths.append(plot_jet_closure(phys, m, outdir))
                if "img_true" in phys:
                    paths.append(plot_density_images(phys, m, outdir,
                                                     "lund_plane.png" if phys["kind"] == "lund" else "jet_images.png"))
            if history:
                paths.append(plot_training_history(history, outdir))
            if token_stats:
                paths.append(plot_token_stats(col, m, outdir, reference))
            for method in projections:
                paths.append(plot_latent_projection(col, names, outdir, method, max_points))
        m["plots"] = [p for p in paths if p]
    m["diagnostics_seconds"] = time.time() - t0
    with open(outdir / "metrics.json", "w") as f:
        json.dump(_py(m), f, indent=2)
    if verbose:
        print_summary(m, outdir)
    return m


def read_first_jets(path, n_jets, n_const, feature_keys, pad_value):
    """Read only the first n_jets jets (cheap on large files). Returns X (J, N, F) float32, mask (J, N)."""
    with h5py.File(path, "r") as f:
        c = f["constituents"]
        missing = [k for k in feature_keys if k not in c]
        if missing:
            raise KeyError(f"{path}: constituents/ has no {missing} (available: {list(c.keys())}); "
                           f"pass --feature-keys, or this file does not match the tokenizer")
        X = np.stack([np.asarray(c[k][:n_jets, :n_const], dtype=np.float32) for k in feature_keys], -1)
    return X, helpers_vqvae.padding_mask(X, pad_value)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    helpers_vqvae.add_common_args(p)
    p.add_argument("--tokenizer", required=True, help="tokenizer checkpoint from train_vqvae.py")
    p.add_argument("--data", required=True, help="input .h5 (pre-processed constituents/)")
    p.add_argument("--out", required=True, help="output directory for plots and metrics.json")
    p.add_argument("--n-jets", type=int, default=20000, help="jets to analyse (first N in the file)")
    p.add_argument("--max-points", type=int, default=5000, help="constituents in t-SNE/UMAP/PCA projections")
    p.add_argument("--projections", nargs="*", default=["pca", "tsne"], choices=["pca", "tsne", "umap"])
    p.add_argument("--token-stats", action="store_true", help="also sequence/token statistics")
    p.add_argument("--no-plots", action="store_true", help="metrics only")
    p.add_argument("--no-physical", action="store_true", help="skip the physical-space view")
    o = p.add_argument_group("data settings (default: taken from the checkpoint)")
    o.add_argument("--input_format", choices=["ktdr", "4vec", "relvec"], default=None)
    o.add_argument("--preprocess", default=None, choices=["none", "log", "linear", "logit", "standardize", "logstd",
                                                           "shiftpad"])
    o.add_argument("--pad-value", type=float, default=None)
    o.add_argument("--feature-keys", nargs="+", default=None)
    o.add_argument("--num-constituents", type=int, default=None)
    o.add_argument("--relvec-order", nargs=3, default=list(RELVEC_ORDER), choices=["ptfrac", "deta", "dphi"],
                   help="role of the 3 relvec features (helpers.make_absolute_constituents uses ptfrac dphi deta)")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    import models_vqvae
    tok, ck = models_vqvae.load_tokenizer(args.tokenizer, args.device)
    ta = ck.get("train_args", {})
    input_format = args.input_format or tok.input_format or ta.get("input_format")
    preprocess = args.preprocess if args.preprocess is not None else ta.get("preprocess")
    preprocess = None if preprocess == "none" else preprocess
    feature_keys = args.feature_keys or ta.get("feature_keys") or list(helpers_vqvae.FOURVEC_KEYS)
    n_const = args.num_constituents or ck.get("n_const") or ta.get("num_constituents", 20)
    pad_value = args.pad_value if args.pad_value is not None else tok.pad_value
    print(f"tokenizer {args.tokenizer}: format {input_format}, pre-processing {preprocess}, features {feature_keys}, "
          f"pad value {pad_value:g}, {n_const} constituents")

    X, mask = read_first_jets(args.data, args.n_jets, n_const, feature_keys, pad_value)
    if X.shape[-1] != tok.F:
        raise ValueError(f"data has {X.shape[-1]} features, tokenizer expects {tok.F}")
    col = collect(tok, X, mask, device=args.device)
    run_diagnostics(col, args.out, input_format=input_format, preprocess=preprocess, feature_keys=feature_keys,
                    reference=ck.get("reference"), history=ck.get("history"), token_stats=args.token_stats,
                    plots=not args.no_plots, physical=not args.no_physical, projections=args.projections,
                    max_points=args.max_points, relvec_order=tuple(args.relvec_order))


if __name__ == "__main__":
    main()
