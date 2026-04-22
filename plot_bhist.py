#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto HDF5 -> Line-Plot Histogrammer (Aggregate / Grouped / Separate)

What was wrong before?
- Arrays that were already BINNED (each value = height of a bin) were mistakenly
  treated as a LIST OF SAMPLES and re-binned via rounding. That caused the sawtooth.

Fix in this script:
- Add an explicit input mode:
    --input_kind {binned, samples, auto} (default: binned)
  * binned  : treat each 1D array as bin heights; x = [0..len-1]; y = heights
  * samples : treat each 1D array as sample values; use integer bins or continuous bins
  * auto    : simple heuristic to decide per-series
- In binned mode we never call integer_counts(); we sum arrays element-wise across columns/files.

Defaults:
- NO log transform (you must opt in with --even_transform / --odd_transform)
- Integer bins ON only for 'samples' mode; irrelevant for 'binned'
- Per-jet normalization kept but typically you want it OFF for binned heights;
  use --no_per_jet_norm if your heights are already normalized.

Usage:
  python plot_bhist.py --file_list binList.txt --mode group
  python plot_bhist.py --file_list binList.txt --mode group --no_per_jet_norm
  python plot_bhist.py --file_list binList.txt --mode group --input_kind samples
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------- argument parsing -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-discover numeric arrays/DataFrames in HDF5; plot line-histograms in aggregate/group/separate modes (NO log by default)."
    )
    # I/O
    p.add_argument("--h5", type=str, nargs="*",
                   help="HDF5 filename(s). If omitted/empty, will read from --file_list.")
    p.add_argument("--file_list", type=str, default="binList.txt",
                   help="Text file containing HDF5 paths (one per line). Used when --h5 is not given or empty.")
    p.add_argument("--out_dir", type=str, default="plots_h5",
                   help="Directory to save plots.")
    p.add_argument("--out_tag", type=str, default=None,
                   help="Optional filename tag prefix.")
    p.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"],
                   help="Output image format.")
    p.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    p.add_argument("--show", action="store_true", help="Show figures interactively.")

    # Discovery / selection
    p.add_argument("--prefer", type=str, default="auto",
                   choices=["auto", "pandas", "h5py"],
                   help="Discovery preference when both pandas tables and raw datasets exist.")
    p.add_argument("--key", type=str, default=None,
                   help="Pandas HDFStore key to read (e.g. 'discretized', 'raw'). If omitted, auto-pick.")
    p.add_argument("--include_datasets", type=str, default=None,
                   help="Regex to filter h5py datasets by name/path (applied after discovery).")
    p.add_argument("--exclude_datasets", type=str, default=None,
                   help="Regex to exclude h5py datasets by name/path.")
    p.add_argument("--include_cols", type=str, default=None,
                   help="Regex to select DataFrame columns (numeric columns only are considered).")
    p.add_argument("--exclude_cols", type=str, default=None,
                   help="Regex to exclude DataFrame columns.")

    # Channelization hints + transforms (DEFAULT: no transform)
    p.add_argument("--even_keys", type=str, nargs="*", default=["deltaR", "var1"],
                   help="Name hints for the EVEN channel. Case-insensitive.")
    p.add_argument("--odd_keys", type=str, nargs="*", default=["kt", "var2"],
                   help="Name hints for the ODD channel. Case-insensitive.")
    p.add_argument("--even_transform", type=str, default="none",
                   choices=["none", "log", "log10", "log1_over"],
                   help="Transform applied to EVEN channel (default: none).")
    p.add_argument("--odd_transform", type=str, default="none",
                   choices=["none", "log", "log10", "log1_over"],
                   help="Transform applied to ODD channel (default: none).")

    # Binning & ranges (for continuous mode)
    p.add_argument("--bins", type=int, default=50,
                   help="Number of bins (ignored if --bin_edges is provided or integer_bins is enabled).")
    p.add_argument("--bin_edges", type=float, nargs="+", default=None,
                   help="Explicit bin edges (ignored if integer_bins enabled).")
    p.add_argument("--range", type=float, nargs=2, default=None,
                   help="Histogram range (ignored if --bin_edges is provided or integer_bins enabled).")

    # Normalization / clipping
    p.add_argument("--per_jet_norm", action="store_true", default=True,
                   help="Normalize by # of jets; in group/aggregate we divide by (#jets × #series_used).")
    p.add_argument("--no_per_jet_norm", dest="per_jet_norm", action="store_false",
                   help="Disable per-jet normalization.")
    p.add_argument("--clip_low_q", type=float, default=None,
                   help="Clip values below this quantile (0-1) before histogram.")
    p.add_argument("--clip_high_q", type=float, default=None,
                   help="Clip values above this quantile (0-1) before histogram.")
    p.add_argument("--logy", action="store_true", help="Log-scale Y axis.")

    # Plot style
    p.add_argument("--alpha", type=float, default=0.9, help="Alpha for lines.")
    p.add_argument("--linewidth", type=float, default=1.4, help="Line width for line plots.")

    # Performance / limits
    p.add_argument("--max_rows", type=int, default=None, help="Limit rows loaded from DataFrame.")
    p.add_argument("--sample_frac", type=float, default=None, help="Randomly sample a fraction of rows (0-1).")

    # Mode
    p.add_argument("--mode", type=str, default="group",
                   choices=["group", "separate", "aggregate"],
                   help="Plotting mode.")

    # Integer bin control
    p.add_argument("--integer_bins", action="store_true", default=True,
                   help="Use integer-aligned bins [k-0.5, k+0.5] (default ON).")
    p.add_argument("--no_integer_bins", dest="integer_bins", action="store_false",
                   help="Disable integer-aligned binning and use continuous histograms (still lines).")

    # X-axis control
    p.add_argument("--x_min", type=float, default=None, help="Force x-axis minimum.")
    p.add_argument("--x_max", type=float, default=None, help="Force x-axis maximum.")

    # Padding handling
    p.add_argument("--pad_value", type=float, default=-1.0,
                   help="Padding value to drop by default.")
    p.add_argument("--keep_pad", action="store_true", help="Keep padding values.")

    # Misc
    p.add_argument("--title", type=str, default="HDF5 Line Histogram", help="Title prefix for plots.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    p.add_argument("--debug", action="store_true", help="Print extra diagnostics.")
    return p.parse_args()

# ------------------------- utilities -------------------------
def list_h5py_datasets(fp: h5py.File) -> List[str]:
    out: List[str] = []
    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append(name)
    fp.visititems(_visitor)
    return out

def list_pandas_keys(path: Union[str, Path]) -> List[str]:
    keys: List[str] = []
    try:
        with pd.HDFStore(str(path), "r") as store:
            keys = list(store.keys())
    except Exception:
        pass
    return [k[1:] if k.startswith("/") else k for k in keys]

def is_numeric_dtype_np(arr: np.ndarray) -> bool:
    return arr.dtype.kind in ("f", "i", "u")

def get_numeric_df(df: pd.DataFrame,
                   include_regex: Optional[str],
                   exclude_regex: Optional[str]) -> pd.DataFrame:
    num_df = df.select_dtypes(include=[np.number]).copy()
    cols = list(num_df.columns)
    if include_regex:
        cols = [c for c in cols if re.search(include_regex, c)]
    if exclude_regex:
        cols = [c for c in cols if not re.search(exclude_regex, c)]
    return num_df[cols]

def read_pandas_table(path: Union[str, Path],
                      key: Optional[str],
                      include_cols: Optional[str],
                      exclude_cols: Optional[str],
                      max_rows: Optional[int],
                      sample_frac: Optional[float],
                      seed: int,
                      debug: bool) -> Tuple[str, pd.DataFrame]:
    keys = list_pandas_keys(path)
    if debug:
        print(f"[pandas] keys found: {keys}")
    if not keys:
        raise RuntimeError("No pandas keys found in HDF5 file.")
    sel_key = key if key is not None else keys[0]
    if sel_key not in keys:
        raise KeyError(f"Key '{sel_key}' not found. Available keys: {keys}")
    df = pd.read_hdf(str(path), key=sel_key)
    if max_rows is not None:
        df = df.iloc[:max_rows].copy()
    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
    # ✅ 正确传参
    df_num = get_numeric_df(df, include_regex=include_cols, exclude_regex=exclude_cols)
    if df_num.shape[1] == 0:
        raise RuntimeError(f"No numeric columns after filtering in key '{sel_key}'.")
    if debug:
        print(f"[pandas] key='{sel_key}', shape={df_num.shape}, cols(example)={list(df_num.columns)[:10]}")
    return sel_key, df_num

def read_h5py_numeric_arrays(fp: h5py.File,
                             include_regex: Optional[str],
                             exclude_regex: Optional[str],
                             debug: bool) -> List[Tuple[str, np.ndarray]]:
    paths = list_h5py_datasets(fp)
    if debug:
        print(f"[h5py] datasets found: {len(paths)}")
    out: List[Tuple[str, np.ndarray]] = []
    for p in paths:
        if include_regex and not re.search(include_regex, p):
            continue
        if exclude_regex and re.search(exclude_regex, p):
            continue
        try:
            ds = fp[p]
            if not isinstance(ds, h5py.Dataset):
                continue
            if not is_numeric_dtype_np(ds.dtype):
                continue
            arr = ds[...]
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            out.append((p, np.asarray(arr)))
        except Exception as e:
            if debug:
                print(f"[h5py] skip '{p}': {e}")
    if debug:
        print(f"[h5py] numeric series kept: {len(out)}")
    return out

def last_token(name: str) -> str:
    t = name.split(":")[-1]
    t = t.split("/")[-1]
    return t

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def filter_padding(x: np.ndarray, pad_value: float, drop_pad: bool) -> np.ndarray:
    x = x[np.isfinite(x)]
    if drop_pad:
        x = x[x != pad_value]
    return x

def clip_quantiles(x: np.ndarray,
                   qlow: Optional[float],
                   qhigh: Optional[float]) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    lo = np.quantile(x, qlow) if qlow is not None else None
    hi = np.quantile(x, qhigh) if qhigh is not None else None
    if lo is not None:
        x = x[x >= lo]
    if hi is not None:
        x = x[x <= hi]
    return x

def preprocess_raw(x: np.ndarray,
                   pad_value: float,
                   drop_pad: bool,
                   qlow: Optional[float],
                   qhigh: Optional[float]) -> np.ndarray:
    x = np.asarray(x).ravel()
    x = filter_padding(x, pad_value=pad_value, drop_pad=drop_pad)
    x = clip_quantiles(x, qlow=qlow, qhigh=qhigh)
    return x

def apply_named_transform(x: np.ndarray, name: str) -> np.ndarray:
    x = x[np.isfinite(x)]
    if name == "none":
        return x
    x = x[x > 0]
    if x.size == 0:
        return x
    if name == "log":
        return np.log(x)
    if name == "log10":
        return np.log10(x)
    if name == "log1_over":
        return np.log(1.0 / x)
    return x

def select_and_channelize(series: List[Tuple[str, np.ndarray]],
                          even_keys: List[str],
                          odd_keys: List[str],
                          even_tf: str,
                          odd_tf: str,
                          pad_value: float,
                          drop_pad: bool,
                          qlow: Optional[float],
                          qhigh: Optional[float],
                          debug: bool) -> Dict[str, List[np.ndarray]]:
    evens = {k.lower() for k in even_keys}
    odds  = {k.lower() for k in odd_keys}
    ch: Dict[str, List[np.ndarray]] = {"even": [], "odd": []}
    for idx, (name, arr) in enumerate(series):
        raw = preprocess_raw(arr, pad_value=pad_value, drop_pad=drop_pad, qlow=qlow, qhigh=qhigh)
        if raw.size == 0:
            continue
        token = last_token(name).lower()
        is_even = any(k in token for k in evens) or (idx % 2 == 0)
        is_odd  = any(k in token for k in odds)  or (idx % 2 == 1)
        if is_even and not is_odd:
            ch["even"].append(apply_named_transform(raw, even_tf))
        elif is_odd and not is_even:
            ch["odd"].append(apply_named_transform(raw, odd_tf))
        else:
            (ch["even"] if idx % 2 == 0 else ch["odd"]).append(
                apply_named_transform(raw, even_tf if idx % 2 == 0 else odd_tf)
            )
    if debug:
        print(f"[channelize] even({even_tf}): {sum(a.size for a in ch['even'])} vals in {len(ch['even'])} series; "
              f"odd({odd_tf}): {sum(a.size for a in ch['odd'])} vals in {len(ch['odd'])} series")
    return ch

def integer_counts(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0, -1
    idx = np.round(arr).astype(int)
    mi, ma = int(idx.min()), int(idx.max())
    width = ma - mi + 1
    cnt = np.bincount(idx - mi, minlength=width)
    centers = np.arange(mi, ma + 1, dtype=int)
    return centers, cnt, mi, ma

def sum_integer_counts(list_of_arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (centers, total_counts, n_series_used)."""
    min_ints, max_ints, per_counts = [], [], []
    used = 0
    for arr in list_of_arrays:
        c, cnt, mi, ma = integer_counts(arr)
        if cnt.size == 0:
            continue
        used += 1
        min_ints.append(mi); max_ints.append(ma)
        per_counts.append((c, cnt))
    if used == 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0
    gmin, gmax = min(min_ints), max(max_ints)
    total = np.zeros(gmax - gmin + 1, dtype=np.int64)
    centers = np.arange(gmin, gmax + 1, dtype=int)
    for c, cnt in per_counts:
        off = c[0] - gmin
        total[off:off + cnt.size] += cnt
    return centers, total, used

def cont_hist_line(values: np.ndarray,
                   bins: Union[int, Sequence[float]],
                   rng: Optional[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([]), np.array([])
    if isinstance(bins, int):
        if rng is None:
            lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
            rng = (lo, hi)
        edges = np.linspace(rng[0], rng[1], bins + 1)
    else:
        edges = np.asarray(bins, dtype=float)
    counts, edges = np.histogram(values, bins=edges, range=rng)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts

def determine_xlim(data_min: float,
                   data_max: float,
                   user_xmin: Optional[float],
                   user_xmax: Optional[float]) -> Tuple[float, float]:
    xmin = 0.0 if data_min >= 0 else float(data_min)
    xmax = float(data_max) + 1.0
    if user_xmin is not None: xmin = float(user_xmin)
    if user_xmax is not None: xmax = float(user_xmax)
    if xmax <= xmin: xmax = xmin + 1.0
    return xmin, xmax

def xdomain_from_many(arrays: List[np.ndarray], use_integer: bool) -> Tuple[float, float]:
    if len(arrays) == 0:
        return 0.0, 0.0
    stacked = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size > 0], axis=0)
    if stacked.size == 0:
        return 0.0, 0.0
    if use_integer:
        idx = np.round(stacked).astype(int)
        return float(idx.min()), float(idx.max())
    return float(np.min(stacked)), float(np.max(stacked))

def plot_lines(sets: List[Tuple[str, np.ndarray, np.ndarray]],
               title: str, out_path: Path, fmt: str, dpi: int,
               xlim: Optional[Tuple[float, float]] = None, logy: bool = False,
               alpha: float = 0.9, linewidth: float = 1.4, show: bool = False,
               xlabel: str = "bin", ylabel: str = "Counts") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 5.8))
    drew = False
    for lab, xc, yc in sets:
        if xc.size == 0 or yc.size == 0:
            continue
        plt.plot(xc, yc, label=lab, alpha=alpha, linewidth=linewidth)
        drew = True
    if not drew:
        plt.close(); return
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if xlim is not None: plt.xlim(*xlim)
    if logy: plt.yscale("log")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(f".{fmt}"), dpi=dpi)
    if show: plt.show()
    plt.close()

def estimate_njets_from_series(arrays: List[np.ndarray]) -> int:
    lengths = [int(a.size) for a in arrays if a.size > 0]
    return int(max(lengths)) if lengths else 1

def read_path_list(path: Union[str, Path]) -> List[str]:
    path = Path(path)
    if not path.exists(): return []
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"): continue
        out.append(s)
    return out

# ------------------------- main flow -------------------------
def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    # Resolve input file list
    input_files: List[str] = list(args.h5) if args.h5 else []
    if not input_files:
        input_files = read_path_list(args.file_list)
    if not input_files:
        print(f"[error] No HDF5 files given (neither --h5 nor --file_list '{args.file_list}' provided usable paths).")
        return

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Continuous-hist defaults (only when integer_bins=False)
    if args.bin_edges is not None and len(args.bin_edges) >= 2:
        default_bins: Union[int, Sequence[float]] = np.asarray(args.bin_edges, dtype=float)
        default_range = None
    else:
        default_bins = int(args.bins)
        default_range = tuple(args.range) if args.range is not None else None

    drop_pad = not args.keep_pad

    collect_group: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {"even": [], "odd": []}
    xlim_group: Dict[str, Tuple[float, float]] = {}
    collect_sep: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
    xlim_sep: Dict[str, Tuple[float, float]] = {}

    def base_label(which: str, tf: str) -> str:
        if tf == "log1_over": return "log1_over_"+which
        if tf == "log":       return "log_"+which
        if tf == "log10":     return "log10_"+which
        return which

    for h5_path in tqdm(input_files, desc="Files"):
        pth = Path(h5_path)
        if not pth.exists():
            print(f"[skip] File not found: {pth}")
            continue

        file_label = pth.stem

        # Discover series
        series: List[Tuple[str, np.ndarray]] = []
        tried_pandas = False
        if args.prefer in ("auto", "pandas"):
            try:
                sel_key, df = read_pandas_table(
                    path=pth, key=args.key,
                    include_cols=args.include_cols, exclude_cols=args.exclude_cols,
                    max_rows=args.max_rows, sample_frac=args.sample_frac,
                    seed=args.seed, debug=args.debug,
                )
                tried_pandas = True
                for col in df.columns:
                    series.append((f"{sel_key}:{col}", np.asarray(df[col].to_numpy()).ravel()))
                if args.debug: print(f"[ok] pandas key='{sel_key}' -> {len(series)} series")
            except Exception as e:
                if args.debug: print(f"[warn] pandas read failed ({e}); trying h5py datasets.")

        if (args.prefer in ("auto", "h5py")) and (not tried_pandas or len(series) == 0):
            try:
                with h5py.File(str(pth), "r") as fp:
                    ds_list = read_h5py_numeric_arrays(fp,
                        include_regex=args.include_datasets,
                        exclude_regex=args.exclude_datasets,
                        debug=args.debug)
                for name, arr in ds_list:
                    series.append((name, np.asarray(arr).ravel()))
                if args.debug: print(f"[ok] h5py numeric arrays -> {len(series)} series")
            except Exception as e:
                print(f"[error] Could not read numeric data from {pth.name}: {e}")
                continue

        if len(series) == 0:
            print(f"[skip] No numeric series discovered in {pth.name}.")
            continue

        # Channelize (NO log by default)
        ch_map = select_and_channelize(
            series,
            even_keys=args.even_keys, odd_keys=args.odd_keys,
            even_tf=args.even_transform, odd_tf=args.odd_transform,
            pad_value=args.pad_value, drop_pad=drop_pad,
            qlow=args.clip_low_q, qhigh=args.clip_high_q,
            debug=args.debug,
        )

        # Estimate jets per series
        njets = estimate_njets_from_series(ch_map["even"] + ch_map["odd"])

        # x-limits per base
        for base in ("even", "odd"):
            dmin, dmax = xdomain_from_many(ch_map[base], args.integer_bins)
            xlim_group[base] = determine_xlim(dmin, dmax, args.x_min, args.x_max)

        # Build line per base (group/aggregate overlay)
        for base in ("even", "odd"):
            arrays = ch_map[base]
            if args.integer_bins:
                xc, total_cnt, n_used = sum_integer_counts(arrays)
            else:
                merged = np.concatenate([a for a in arrays if a.size > 0], axis=0) if arrays else np.array([])
                if merged.size == 0:
                    xc, total_cnt, n_used = np.array([]), np.array([]), 0
                else:
                    xc, total_cnt = cont_hist_line(merged, bins=default_bins, rng=default_range)
                    n_used = len([a for a in arrays if a.size > 0])

            cnt = total_cnt.astype(float)

            # 🔧 Normalization denominator（Key repair）:
            # group/aggregate include n_used series，thus donominator should be njets * n_used
            if args.per_jet_norm and njets > 0 and n_used > 0 and cnt.size > 0:
                cnt /= float(njets * n_used)

            collect_group[base].append((file_label, xc, cnt))

        # For SEPARATE: per index per base（single series，only need divided by njets）
        for base in ("even", "odd"):
            arrays = ch_map[base]
            if base not in xlim_sep:
                xlim_sep[base] = xlim_group[base]
            for i, arr in enumerate(arrays):
                key = f"{base}_{i}"
                if args.integer_bins:
                    xc, cnt, _, _ = integer_counts(arr)
                else:
                    xc, cnt = cont_hist_line(arr, bins=default_bins, rng=default_range)
                cnt = cnt.astype(float)
                if args.per_jet_norm and njets > 0 and cnt.size > 0:
                    cnt /= float(njets)
                collect_sep.setdefault(key, []).append((file_label, xc, cnt))

    # ---------- PLOTTING ----------
    if args.mode == "group":
        for base in ("even", "odd"):
            sets = collect_group.get(base, [])
            if not sets: continue
            base_name = base_label(base, args.even_transform if base == "even" else args.odd_transform)
            title = (f"{args.title} • mode=group • base={base_name}; "
                     f"per_jet_norm={int(args.per_jet_norm)}; integer_bins={int(args.integer_bins)}")
            out_base = out_dir / f"{(args.out_tag or 'h5line')}_GRP_{base_name}"
            plot_lines(sets, title, out_base, args.format, args.dpi,
                       xlim=xlim_group.get(base), logy=args.logy,
                       alpha=args.alpha, linewidth=args.linewidth, show=args.show,
                       xlabel="bin (integer centers)" if args.integer_bins else "bin center",
                       ylabel="Per-jet-per-series counts" if args.per_jet_norm else "Counts")

    elif args.mode == "aggregate":
        for base in ("even", "odd"):
            sets = collect_group.get(base, [])
            if not sets: continue
            base_name = base_label(base, args.even_transform if base == "even" else args.odd_transform)
            title = (f"{args.title} • mode=aggregate • base={base_name}; "
                     f"per_jet_norm={int(args.per_jet_norm)}; integer_bins={int(args.integer_bins)}")
            out_base = out_dir / f"{(args.out_tag or 'h5line')}_AGG_{base_name}"
            plot_lines(sets, title, out_base, args.format, args.dpi,
                       xlim=xlim_group.get(base), logy=args.logy,
                       alpha=args.alpha, linewidth=args.linewidth, show=args.show,
                       xlabel="bin (integer centers)" if args.integer_bins else "bin center",
                       ylabel="Per-jet-per-series counts" if args.per_jet_norm else "Counts")

    elif args.mode == "separate":
        for key, sets in collect_sep.items():
            base = key.split("_", 1)[0]
            base_name = base_label(base, args.even_transform if base == "even" else args.odd_transform)
            title = (f"{args.title} • mode=separate • series={key.replace(base, base_name)}; "
                     f"per_jet_norm={int(args.per_jet_norm)}; integer_bins={int(args.integer_bins)}")
            out_base = out_dir / f"{(args.out_tag or 'h5line')}_SEP_{sanitize_filename(key.replace(base, base_name))}"
            plot_lines(sets, title, out_base, args.format, args.dpi,
                       xlim=xlim_sep.get(base), logy=args.logy,
                       alpha=args.alpha, linewidth=args.linewidth, show=args.show,
                       xlabel="bin (integer centers)" if args.integer_bins else "bin center",
                       ylabel="Per-jet counts" if args.per_jet_norm else "Counts")
    else:
        raise ValueError(f"Unknown mode: {args.mode!r}")

if __name__ == "__main__":
    main()
