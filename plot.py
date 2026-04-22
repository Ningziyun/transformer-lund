#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 20:30:51 2025

@author: ningyan
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare kt/deltaR or pt/eta/phi distributions across multiple ROOT files,
and also draw per-file 2D Lund-plane histograms with smart (rich) titles.

Highlights
----------
- Title params layout:
    --title_params_layout {one_line,multi_line}  (default: one_line)
    --title_param_sep "; "   # separator used when one_line
- Auto mark differences across many files:
    (2nd line: file stem in red by default; optional 3rd-line tokens via --mark_line3_diff)
- Rich title mode (--title_rich) with per-segment color/size/weight and safe auto-fit.

Requirements:
  pip install uproot awkward numpy matplotlib tqdm
"""

import argparse
from pathlib import Path
import os
import re
from typing import Optional, List, Dict, Tuple

import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from tqdm import tqdm


# =========================
# Argument parsing
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 2D Lund-plane histograms from ROOT TTrees, one plot per file."
    )
    # I/O
    parser.add_argument("--file_list", type=str, default="fileList.txt",
                        help="Text file listing ROOT files (one path per line).")
    parser.add_argument("--tree_name", type=str, default="auto",
                        help="TTree name; use 'auto' to detect per file (searches subdirectories).")

    # Semantic interpretation of the FIRST and SECOND selected branches
    parser.add_argument("--first_kind", type=str, default="logDRinv",
                        choices=["logDRinv", "logkt", "deltaR", "kt"],
                        help="Semantic of the FIRST picked branch (default: logDRinv).")
    parser.add_argument("--second_kind", type=str, default="logkt",
                        choices=["logDRinv", "logkt", "deltaR", "kt"],
                        help="Semantic of the SECOND picked branch (default: logkt).")

    # Output
    parser.add_argument("--out_dir", type=str, default="plots",
                        help="Directory to save outputs.")
    parser.add_argument("--out_tag", type=str, default="lund2d",
                        help="Output filename prefix. Final filename is <out_tag>_<file-stem>.(png|npz).")

    # Binning & ranges
    parser.add_argument("--xbins", type=int, default=50, help="Number of X bins.")
    parser.add_argument("--ybins", type=int, default=50, help="Number of Y bins.")
    parser.add_argument("--xrange", type=float, nargs=2, default=None,
                        help="X range: xmin xmax (default auto).")
    parser.add_argument("--yrange", type=float, nargs=2, default=None,
                        help="Y range: ymin ymax (default auto).")

    # Normalization mode
    parser.add_argument("--norm", type=str, default="per_jet",
                        choices=["none", "per_jet", "per_emission"],
                        help="Z normalization: 'none' (counts), 'per_jet' (counts/#jets), "
                             "'per_emission' (counts / total counts in-range). Default: per_jet.")

    # Z-axis (colorbar) range
    parser.add_argument("--zmin", type=float, default=0.0,
                        help="Colorbar lower bound. Default 0.")
    parser.add_argument("--zmax", type=float, default=None,
                        help="Colorbar upper bound. Default auto (max).")

    # Plot setting
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Matplotlib colormap name, e.g. 'viridis', 'plasma', 'magma'.")

    # Plot title (basic)
    parser.add_argument("--title", type=str, default="QCD Lund Plane",
                        help="Base title (1st line).")
    parser.add_argument("--title_extra", type=str, default="",
                        help="Optional note appended among the parameter tokens (e.g., 'epoch=10').")
    parser.add_argument("--title_size", type=float, default=None,
                        help="Default font size for normal/rich titles (used as base size).")

    # Auto-mark differences in title (default ON)
    parser.add_argument("--mark_diff", dest="mark_diff", action="store_true", default=True,
                        help="Highlight differing parts across files (default: on).")
    parser.add_argument("--no_mark_diff", dest="mark_diff", action="store_false",
                        help="Disable highlighting differences in titles.")
    parser.add_argument("--diff_color", type=str, default="red",
                        help="Color used to highlight differing parts (default: red).")
    parser.add_argument("--mark_line3_diff", action="store_true",
                        help="Also color parameter tokens that truly vary across files.")

    # >>> NEW: control how parameter tokens are laid out
    parser.add_argument("--title_params_layout", type=str,
                        choices=["one_line", "multi_line"], default="one_line",
                        help="How to place parameter tokens: "
                             "'one_line' keeps all tokens on the 3rd line; "
                             "'multi_line' puts one token per line (lines 3,4,5,...)")
    parser.add_argument("--title_param_sep", type=str, default="; ",
                        help="Separator between tokens for 'one_line' layout (default: '; ').")

    # Rich title controls (advanced)
    parser.add_argument("--title_rich", type=str, default=None,
                        help=("Rich title: segments separated by ';'. "
                              "Each segment is 'text|color|size|weight'. "
                              "Use '\\n' or '<br>' (as a standalone segment) to break line."))
    parser.add_argument("--title_fig_y", type=float, default=0.985,
                        help="Vertical anchor (figure coords) for rich title top (0~1).")
    parser.add_argument("--title_seg_sep", type=float, default=0.0,
                        help="Horizontal spacing between segments on the same line.")
    parser.add_argument("--title_line_sep", type=float, default=2.0,
                        help="Vertical spacing between lines in rich title.")
    parser.add_argument("--title_width_max_frac", type=float, default=0.95,
                        help="Max allowed title width fraction of figure width before downscaling.")
    parser.add_argument("--title_top_pad_frac", type=float, default=0.015,
                        help="Extra top padding fraction reserved above axes.")
    parser.add_argument("--title_min_size", type=float, default=6.0,
                        help="Lower bound for autoscaled font size.")

    # Per-jet emission truncation
    parser.add_argument("--max_emissions_per_jet", "--maxN", type=int, default=None,
                        help="If set, truncate each jet's emissions to the FIRST N per jet.")
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true",
                        help="Parse and report counts; do not produce plots.")
    parser.add_argument("--debug", action="store_true",
                        help="Print extra debug info about branches.")

    return parser.parse_args()


# =========================
# Utilities (I/O & picking)
# =========================
def read_file_list(path_txt: str) -> list:
    p = Path(path_txt)
    if not p.exists():
        raise FileNotFoundError(f"file_list not found: {p}")
    files = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        files.append(s)
    if not files:
        raise RuntimeError(f"No valid ROOT file paths found in {p}")
    return files


def _collect_ttrees(node) -> list:
    trees = []
    for k, cls in node.classnames().items():
        base = k.split(";")[0]
        obj = node[k]
        if cls == "TTree":
            trees.append(base)
        elif cls in ("TDirectoryFile", "TDirectory"):
            subt = _collect_ttrees(obj)
            trees.extend([f"{base}/{t}" for t in subt])
    return trees


def _pick_tree_for_file(uf, explicit_tree):
    if explicit_tree and explicit_tree != "auto":
        try:
            return uf[explicit_tree], explicit_tree
        except Exception:
            base = explicit_tree.split(";")[0]
            return uf[base], explicit_tree
    trees = _collect_ttrees(uf)
    if not trees:
        raise KeyError("No TTree found in file.")
    tpath = trees[0]
    return uf[tpath], tpath


def _is_array_like_branch(tree, bname: str, sample_events: int = 64) -> bool:
    # Practical numeric (flat or jagged) test.
    try:
        n = getattr(tree, "num_entries", 0) or 0
        entry_stop = min(n, sample_events) if n > 0 else None
        arr = tree[bname].array(library="ak", entry_stop=entry_stop)
        flat = ak.flatten(arr, axis=None)
        if flat is None:
            return False
        if len(flat) == 0:
            return True
        np_arr = ak.to_numpy(flat)
        return np_arr.dtype is not None and np_arr.dtype.kind in ("f", "i", "u")
    except Exception:
        return False


def _debug_branch_signature(tree, limit=50) -> str:
    items = []
    names = list(tree.keys())
    for b in names[:limit]:
        try:
            tname = tree[b].typename
        except Exception:
            tname = "<?>"
        items.append(f"{b} : {tname}")
    more = "" if len(names) <= limit else f" ... (+{len(names)-limit} more)"
    return "\n".join(items) + more


def _name_priority_score(name: str) -> int:
    s = name.lower()
    score = 0
    if re.search(r"\bkt\b", s) or "logkt" in s:
        score += 5
    if "deltar" in s or "delta" in s or re.search(r"\bdr\b", s):
        score += 5
    if "log" in s:
        score += 2
    if "lund" in s:
        score += 1
    return score


def _pick_first_two_array_branches(tree, debug=False) -> list:
    names = list(tree.keys())
    good = [b for b in names if _is_array_like_branch(tree, b)]
    if len(good) >= 2:
        picked = good[:2]
        if debug:
            print(f"[debug] sample-picked branches: {picked}")
        return picked
    ranked = sorted(names, key=lambda n: _name_priority_score(n), reverse=True)
    picked = []
    for b in ranked:
        if _is_array_like_branch(tree, b):
            picked.append(b)
            if len(picked) == 2:
                if debug:
                    print(f"[debug] heuristic-picked branches: {picked}")
                return picked
    sig = _debug_branch_signature(tree)
    raise RuntimeError(
        "Could not find two array-like (numeric) branches to read.\n"
        "Here are some branches and types (first 50):\n"
        f"{sig}\n"
        "Tip: pass an explicit --tree_name or export numeric arrays."
    )


# =========================
# Array transforms
# =========================
def _read_two_branches_as_arrays(tree, b1, b2):
    a1 = tree[b1].array(library="ak")
    a2 = tree[b2].array(library="ak")
    return a1, a2


def _truncate_per_jet(a: ak.Array, maxN: Optional[int]) -> ak.Array:
    if maxN is None:
        return a
    if maxN <= 0:
        raise ValueError("--max_emissions_per_jet must be positive if provided.")
    try:
        _ = ak.num(a, axis=1)
    except Exception:
        return a
    return a[:, :maxN]


def _interpret_and_to_logs(first_flat, second_flat, first_kind, second_kind):
    X = None  # log(1/dR)
    Y = None  # log(kt)
    eps = 1e-12

    def to_logDRinv(x):
        x = np.clip(x, eps, None)
        return np.log(1.0 / x)

    def to_logkt(y):
        y = np.clip(y, eps, None)
        return np.log(y)

    if first_kind == "logDRinv":
        X = first_flat
    elif first_kind == "logkt":
        Y = first_flat
    elif first_kind == "deltaR":
        X = to_logDRinv(first_flat)
    elif first_kind == "kt":
        Y = to_logkt(first_flat)

    if second_kind == "logDRinv":
        X = second_flat if X is None else X
    elif second_kind == "logkt":
        Y = second_flat if Y is None else Y
    elif second_kind == "deltaR":
        X = to_logDRinv(second_flat) if X is None else X
    elif second_kind == "kt":
        Y = to_logkt(second_flat) if Y is None else Y

    if X is None or Y is None:
        raise ValueError("Ambiguous kinds: could not derive both X=log(1/ΔR) and Y=log(kt).")
    return X, Y


# =========================
# Rich title helpers
# =========================
def _build_title_box_from_lines(lines: List[List[Dict]],
                                seg_sep: float, line_sep: float,
                                default_size: Optional[float],
                                size_scale: float = 1.0,
                                min_size: float = 6.0):
    """Build an OffsetBox directly from structured lines/segments."""
    def seg_textarea(seg: Dict):
        props = {}
        if seg.get("color"):  props["color"] = seg["color"]
        if seg.get("weight"): props["weight"] = seg["weight"]
        fsz = seg.get("size", default_size)
        if fsz is not None:
            props["size"] = max(min_size, float(fsz) * size_scale)
        return TextArea(seg.get("text", ""), textprops=props)

    line_boxes = []
    for line in lines:
        items = [seg_textarea(seg) for seg in line]
        line_boxes.append(HPacker(children=items, align="center", pad=0, sep=seg_sep))
    return line_boxes[0] if len(line_boxes) == 1 else VPacker(children=line_boxes, align="center", pad=0, sep=line_sep)


def _parse_rich_spec(spec: str, default_size: Optional[float]) -> List[List[Dict]]:
    """
    Parse 'text|color|size|weight' segments into lines of segment dicts.
    IMPORTANT: Only '\\n' or '<br>' are treated as explicit newlines (empty token is NOT a newline).
    """
    def is_newline(tok: str) -> bool:
        t = tok.strip().lower()
        return t in ("\\n", "<br>")

    lines, cur = [], []
    for raw in spec.split(";"):
        tok = raw  # do not strip here to preserve leading/trailing spaces if intended
        if is_newline(tok):
            if cur or not lines:
                lines.append(cur)
                cur = []
            continue
        parts = tok.split("|")
        text = parts[0] if len(parts) > 0 else ""
        color = parts[1] if len(parts) > 1 and parts[1] else None
        try:
            size = float(parts[2]) if len(parts) > 2 and parts[2] else default_size
        except Exception:
            size = default_size
        weight = parts[3] if len(parts) > 3 and parts[3] else None
        cur.append({"text": text, "color": color, "size": size, "weight": weight})
    if cur:
        lines.append(cur)
    return lines


def _add_rich_title_lines(fig,
                          lines: List[List[Dict]],
                          default_size: Optional[float],
                          y: float,
                          seg_sep: float,
                          line_sep: float,
                          width_max_frac: float,
                          top_pad_frac: float,
                          min_size: float):
    """
    Render a rich title from pre-built line/segment structures (no string parsing),
    auto-fitting width and reserving top margin.
    """
    if not lines:
        return
    box = _build_title_box_from_lines(lines, seg_sep, line_sep, default_size, size_scale=1.0, min_size=min_size)
    anch = AnchoredOffsetbox(loc="upper center", child=box, pad=0.0,
                             bbox_to_anchor=(0.5, y), bbox_transform=fig.transFigure, frameon=False)
    fig.add_artist(anch)
    fig.canvas.draw()
    bb = box.get_window_extent(fig.canvas.get_renderer())
    fig_w, fig_h = fig.bbox.width, fig.bbox.height
    width_frac = bb.width / fig_w
    if width_frac > width_max_frac:
        scale = width_max_frac / max(width_frac, 1e-9)
        fig.artists.remove(anch)
        box = _build_title_box_from_lines(lines, seg_sep, line_sep, default_size,
                                          size_scale=scale, min_size=min_size)
        anch = AnchoredOffsetbox(loc="upper center", child=box, pad=0.0,
                                 bbox_to_anchor=(0.5, y), bbox_transform=fig.transFigure, frameon=False)
        fig.add_artist(anch)
        fig.canvas.draw()
        bb = box.get_window_extent(fig.canvas.get_renderer())
    height_frac = bb.height / fig_h
    top_rect = max(0.0, 1.0 - height_frac - top_pad_frac)
    plt.tight_layout(rect=[0, 0, 1, top_rect])


def _add_rich_title_spec(fig,
                         spec: str,
                         default_size: Optional[float],
                         y: float,
                         seg_sep: float,
                         line_sep: float,
                         width_max_frac: float,
                         top_pad_frac: float,
                         min_size: float):
    """Render a rich title from a spec string (kept for --title_rich)."""
    lines = _parse_rich_spec(spec, default_size)
    _add_rich_title_lines(fig, lines, default_size, y, seg_sep, line_sep,
                          width_max_frac, top_pad_frac, min_size)


# =========================
# 2D histogram + saving
# =========================
def plot_2d_hist(
    x, y, n_jets, xbins, ybins, xrange_, yrange_,
    title_payload, use_rich_lines, fig_args, cmap, out_png, out_npz, norm_mode="per_jet",
    zmin=0.0, zmax=None
):
    """
    Build a 2D histogram H(x,y). Normalize and save.
    Title:
      - if use_rich_lines=True: `title_payload` is List[List[Dict]] lines.
      - else: `title_payload` is a rich-spec string (for --title_rich) or a plain str.
    """
    if norm_mode == "per_jet" and n_jets <= 0:
        raise ValueError("n_jets must be positive to normalize per jet.")

    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[xbins, ybins],
        range=[xrange_, yrange_] if (xrange_ is not None and yrange_ is not None) else None
    )
    raw_emissions_in_range = H.sum()

    # Normalization
    if norm_mode == "per_jet":
        Hn = H / float(n_jets)
        zlabel = "counts per jet"
    elif norm_mode == "per_emission":
        denom = raw_emissions_in_range
        Hn = H if denom == 0 else H / float(denom)
        zlabel = "probability per emission"
    else:
        Hn = H
        zlabel = "counts"

    # Plot
    fig = plt.figure(figsize=(7.8, 6.6))
    mesh = plt.pcolormesh(
        xedges, yedges, Hn.T,
        shading="auto", cmap=cmap, vmin=zmin, vmax=zmax
    )
    cbar = plt.colorbar(mesh); cbar.set_label(zlabel)
    plt.xlabel("log(1/ΔR)"); plt.ylabel("log(kt)")

    # Title
    if use_rich_lines:
        _add_rich_title_lines(
            fig,
            lines=title_payload,
            default_size=fig_args["title_size"],
            y=fig_args["title_fig_y"],
            seg_sep=fig_args["title_seg_sep"],
            line_sep=fig_args["title_line_sep"],
            width_max_frac=fig_args["title_width_max_frac"],
            top_pad_frac=fig_args["title_top_pad_frac"],
            min_size=fig_args["title_min_size"],
        )
    else:
        if isinstance(title_payload, str) and ("|" in title_payload or "\\n" in title_payload or "<br>" in title_payload):
            _add_rich_title_spec(
                fig,
                spec=title_payload,
                default_size=fig_args["title_size"],
                y=fig_args["title_fig_y"],
                seg_sep=fig_args["title_seg_sep"],
                line_sep=fig_args["title_line_sep"],
                width_max_frac=fig_args["title_width_max_frac"],
                top_pad_frac=fig_args["title_top_pad_frac"],
                min_size=fig_args["title_min_size"],
            )
        else:
            plt.title(str(title_payload))
            plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    # Save arrays
    np.savez(out_npz, H=Hn, xedges=xedges, yedges=yedges,
             norm_mode=norm_mode, raw_emissions_in_range=raw_emissions_in_range)
    return Hn.sum(), raw_emissions_in_range


# =========================
# Main
# =========================
def main():
    args = parse_args()
    np.random.seed(args.seed)

    files = read_file_list(args.file_list)
    print(f"[files] {len(files)} files from {args.file_list}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    multi = len(files) > 1

    # ---------- First pass: collect per-file metadata for "diff" detection ----------
    metas = []
    for f in tqdm(files, desc="Scanning files for title info"):
        with uproot.open(f) as uf:
            tree, _ = _pick_tree_for_file(uf, args.tree_name)
            b1, b2 = _pick_first_two_array_branches(tree, debug=args.debug)
            a1, a2 = _read_two_branches_as_arrays(tree, b1, b2)
            if args.max_emissions_per_jet is not None:
                a1 = _truncate_per_jet(a1, args.max_emissions_per_jet)
                a2 = _truncate_per_jet(a2, args.max_emissions_per_jet)
            n_jets = len(a1)
            first_flat = ak.to_numpy(ak.flatten(a1, axis=None))
            second_flat = ak.to_numpy(ak.flatten(a2, axis=None))
            mask = np.isfinite(first_flat) & np.isfinite(second_flat)
            first_flat = first_flat[mask]; second_flat = second_flat[mask]
            X, Y = _interpret_and_to_logs(first_flat, second_flat, args.first_kind, args.second_kind)
            metas.append({
                "path": f,
                "stem": Path(f).stem,
                "n_jets": n_jets,
                "n_emit": len(X),
                "b1": b1, "b2": b2
            })

    # Which tokens really vary across files?
    varying = {"norm": False, "N_jets": False, "N_emissions": False, "maxN": False, "extra": False}
    if multi:
        if len({m["n_jets"] for m in metas}) > 1: varying["N_jets"] = True
        if len({m["n_emit"] for m in metas}) > 1: varying["N_emissions"] = True
        # run-level options are typically constant in one run
        varying["norm"]  = False
        varying["maxN"]  = False
        varying["extra"] = False if not args.title_extra else False

    # ---------- Second pass: actually plot ----------
    for m in tqdm(metas, desc="Plotting files"):
        f = m["path"]; stem = m["stem"]

        out_tag = f"{args.out_tag}_{stem}" if args.out_tag else f"lund2d_{stem}"
        out_png = str(out_dir / f"{out_tag}.png")
        out_npz = str(out_dir / f"{out_tag}.npz")

        with uproot.open(f) as uf:
            tree, tpath = _pick_tree_for_file(uf, args.tree_name)
            b1, b2 = _pick_first_two_array_branches(tree, debug=False)
            a1, a2 = _read_two_branches_as_arrays(tree, b1, b2)
            if args.max_emissions_per_jet is not None:
                a1 = _truncate_per_jet(a1, args.max_emissions_per_jet)
                a2 = _truncate_per_jet(a2, args.max_emissions_per_jet)

            n_jets = len(a1)
            first_flat = ak.to_numpy(ak.flatten(a1, axis=None))
            second_flat = ak.to_numpy(ak.flatten(a2, axis=None))
            mask = np.isfinite(first_flat) & np.isfinite(second_flat)
            first_flat = first_flat[mask]; second_flat = second_flat[mask]
            X, Y = _interpret_and_to_logs(first_flat, second_flat, args.first_kind, args.second_kind)
            n_emit = len(X)

            if args.dry_run:
                print(f"[dry-run] {Path(f).name}: jets={n_jets}, emissions(after mask)={n_emit}, "
                      f"tree='{tpath}', first='{b1}', second='{b2}', maxN={args.max_emissions_per_jet}")
                continue

            # ----- Build title -----
            fig_args = dict(
                title_size=args.title_size,
                title_fig_y=args.title_fig_y,
                title_seg_sep=args.title_seg_sep,
                title_line_sep=args.title_line_sep,
                title_width_max_frac=args.title_width_max_frac,
                title_top_pad_frac=args.title_top_pad_frac,
                title_min_size=args.title_min_size,
            )

            if args.title_rich:
                # Use user-provided rich spec string verbatim.
                title_payload = args.title_rich
                use_rich_lines = False
            else:
                # Build structured lines so we can safely include separators like '; '
                lines: List[List[Dict]] = []

                # Line 1: base title (bold)
                lines.append([{"text": args.title, "weight": "bold"}])

                # Line 2: file stem (red if multi & mark_diff)
                color_stem = args.diff_color if (multi and args.mark_diff) else None
                lines.append([{"text": stem, "color": color_stem}])

                # Line 3 (or multi lines): parameters
                tokens = []
                tokens.append(("norm",        f"norm={args.norm}"))
                tokens.append(("N_jets",      f"N_jets={n_jets}"))
                tokens.append(("N_emissions", f"N_emissions={n_emit}"))
                if args.max_emissions_per_jet is not None:
                    tokens.append(("maxN", f"maxN={args.max_emissions_per_jet}"))
                if args.title_extra:
                    tokens.append(("extra", args.title_extra))

                if args.title_params_layout == "one_line":
                    line3: List[Dict] = []
                    for i, (key, text) in enumerate(tokens):
                        color = args.diff_color if (args.mark_diff and args.mark_line3_diff and varying.get(key, False)) else None
                        line3.append({"text": text, "color": color})
                        if i != len(tokens) - 1:
                            # separator stays literal (e.g., '; ') and does NOT trigger parsing issues
                            line3.append({"text": args.title_param_sep})
                    lines.append(line3)
                else:
                    # multi_line: one token per line
                    for (key, text) in tokens:
                        color = args.diff_color if (args.mark_diff and args.mark_line3_diff and varying.get(key, False)) else None
                        lines.append([{"text": text, "color": color}])

                title_payload = lines
                use_rich_lines = True

            sum_after_norm, raw_in_range = plot_2d_hist(
                x=X, y=Y, n_jets=n_jets,
                xbins=args.xbins, ybins=args.ybins,
                xrange_=tuple(args.xrange) if args.xrange else None,
                yrange_=tuple(args.yrange) if args.yrange else None,
                title_payload=title_payload,
                use_rich_lines=use_rich_lines,
                fig_args=fig_args,
                cmap=args.cmap,
                out_png=out_png, out_npz=out_npz,
                norm_mode=args.norm, zmin=args.zmin, zmax=args.zmax,
            )

            print(f"[done] {Path(f).name} -> {out_png}")
            print(f"[done] {Path(f).name} -> {out_npz}")
            print(f"[check] sum(H after norm)={sum_after_norm:.6f} ; emissions in-range={int(raw_in_range)} ; "
                  f"branches: '{b1}', '{b2}' ; tree='{tpath}' ; maxN={args.max_emissions_per_jet}")


if __name__ == "__main__":
    main()
