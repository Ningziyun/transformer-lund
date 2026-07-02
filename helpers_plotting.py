import os,sys
import time
#os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", f"matplotlib-{os.environ.get('USER', 'user')}"))
#os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import numpy as np
import math
import csv
import re
import textwrap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch

import helpers

# ---------------------------------------------------------------------
# Support functions
# ---------------------------------------------------------------------
def _parse_caption_fields(label):
    text = str(label).replace("\n", " ")
    fields = {}
    parts = str(label).splitlines()
    if len(parts) > 0:
        first_tokens = parts[0].split()
        if len(first_tokens) > 0:
            fields["model"] = first_tokens[0]

    # Captions for intermediate checkpoints can include both the plotted
    # checkpoint and the run's best epoch, e.g. "model ep 1 best ep 7".
    # The comparison title should describe the plotted stage first.
    match = re.search(r"(?<!best )\bep\s+\d+", text)
    if match:
        fields["checkpoint"] = match.group(0)
    else:
        match = re.search(r"\bbest ep\s+\d+", text)
        if match:
            fields["checkpoint"] = match.group(0)

    for key, pattern, label_text in (
        ("bs", r"\bbs\s+([^\s]+)", "bs"),
        ("lr", r"\blr\s+([^\s]+)", "lr"),
        ("tot", r"\btot\s+([^\s]+)", "tot"),
        ("min_lr", r"\bmin lr\s+([^\s]+)", "min lr"),
        ("lr_f", r"\blr_f\s+([^\s]+)", "lr_f"),
        ("start_ep", r"\bstart ep\s+([^\s]+)", "start ep"),
    ):
        match = re.search(pattern, text)
        if match:
            fields[key] = f"{label_text} {match.group(1)}"

    sched_match = re.search(r"\bsched\s+(.+?)(?:\s+start ep|\s+lr_f|\s+min lr|$)", text)
    if sched_match:
        fields["sched"] = f"sched {sched_match.group(1).strip()}"
    elif "cos damping" in text:
        fields["sched"] = "sched cos damping"
    elif "cosine" in text:
        fields["sched"] = "sched cosine"
    elif "plateau" in text:
        fields["sched"] = "sched plateau"

    return fields

def _caption_comparison(labels, first_run_idx=1):
    run_labels = list(labels[first_run_idx:])
    field_order = ["model", "checkpoint", "bs", "lr", "tot", "sched", "start_ep", "lr_f", "min_lr"]
    parsed = [_parse_caption_fields(label) for label in run_labels]

    common = []
    diff_labels = []
    for idx, fields in enumerate(parsed):
        items = []
        for key in field_order:
            values = [item.get(key) for item in parsed]
            present_values = [value for value in values if value not in (None, "")]
            if len(present_values) == len(parsed) and len(set(present_values)) == 1:
                continue
            if fields.get(key):
                items.append(fields[key])
        diff_labels.append("\n".join(items) if items else f"run {idx + 1}")

    if parsed:
        for key in field_order:
            values = [item.get(key) for item in parsed]
            present_values = [value for value in values if value not in (None, "")]
            if len(present_values) == len(parsed) and len(set(present_values)) == 1:
                common.append(present_values[0])

    return diff_labels, common

def _plot_note_from_common(common_items, unavailable_notes=None):
    note_lines = []
    if common_items:
        note_lines.append("Plot: " + "; ".join(common_items))
    if unavailable_notes:
        note_lines.extend(
            f"{str(label).replace(chr(10), ' ')}: unavailable ({reason})"
            for label, reason in unavailable_notes
        )
    return note_lines

def _wrapped_note(note_lines, width=150):
    if not note_lines:
        return ""
    return "\n".join(textwrap.wrap(" | ".join(note_lines), width=width, break_long_words=False))

# ---------------------------------------------------------------------
# Main plots
# ---------------------------------------------------------------------
def projection_plot(inputs,labels=["original","generated","predicted"],out_dir="./Plots/", name="projection", unavailable_notes=None):

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  linestyles=["-","--","-.",":"]

  #Get ranges
  Ndim=inputs[0].shape[1]
  Nin=len(inputs)
  mins=np.zeros(Ndim)
  maxs=np.zeros(Ndim)
  for ii in range(Ndim):
    for jj in range(Nin):
      mins[ii]=min(mins[ii],np.min(inputs[jj][:,ii]))
      maxs[ii]=max(maxs[ii],np.max(inputs[jj][:,ii]))

  #make plot
  fig, axs = plt.subplots(Ndim,1,figsize=(8.0,8.0))
  if Ndim==1: axs=[axs]
  for ii in range(Ndim):
    for jj in range(Nin):
      axs[ii].hist(inputs[jj][:,ii],bins=20,range=[mins[ii],maxs[ii]],histtype="step",density=False,linestyle=linestyles[jj],label=labels[jj])
    if ii==0: axs[0].legend()
    axs[ii].set_yscale("log")

  if unavailable_notes:
    note_text = "Unavailable:\n" + "\n".join(
      f"{label}: {reason}" for label, reason in unavailable_notes
    )
    fig.text(0.98, 0.5, note_text, ha="right", va="center", fontsize=8)

  fig.savefig(os.path.join(out_dir,name+".png"))
  fig.savefig(os.path.join(out_dir,name+".pdf"))
  plt.close(fig)
  print(f"Plotting projections to {out_dir}/{name}.pdf")

def lund_plot(
    inputs,
    labels=["original","generated","predicted"],
    out_dir="./Plots/",
    hist2d_xrange=None,
    hist2d_yrange=None,
    hist2d_bins=(20, 20),
    hist2d_shape=None,
    hist2d_layout=None,
    unavailable_notes=None,
):

  #Get ranges
  Ndim=inputs[0].shape[1]
  Nin=len(inputs)
  unavailable_notes = unavailable_notes or []
  Nplots = Nin + len(unavailable_notes)
  use_compared_titles = Nplots > 1
  diff_labels, common_items = _caption_comparison(labels, first_run_idx=1) if use_compared_titles else ([], [])

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  mins=np.zeros(Ndim)
  maxs=np.zeros(Ndim)
  for ii in range(Ndim):
    for jj in range(Nin):
      mins[ii]=min(mins[ii],np.min(inputs[jj][:,ii]))
      maxs[ii]=max(maxs[ii],np.max(inputs[jj][:,ii]))

  #FIXME 1=DR 0=kt
  mins[0]=-5
  maxs[0]=7
  mins[1]=-1
  maxs[1]=10

  #Make plot
  if Ndim>=2:
    # Create subplots for each input (original, generated, etc.)
    if hist2d_shape is None:
      hist2d_shape = hist2d_layout
    nrows, ncols = resolve_hist2d_shape(Nplots, hist2d_shape)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols + 1.0, 4.4 * nrows + 0.6),
        squeeze=False,
    )
    flat_axs = axs.ravel()
    used_axs = flat_axs[:Nplots]

    last_hist = None
    for jj in range(Nin):
        ax = used_axs[jj]
        # Determine plotting range
        x_range = hist2d_xrange if hist2d_xrange is not None else [mins[1], maxs[1]]
        y_range = hist2d_yrange if hist2d_yrange is not None else [mins[0], maxs[0]]

        # Plot 2D histogram
        last_hist = ax.hist2d(
            inputs[jj][:, 1],
            inputs[jj][:, 0],
            range=[x_range, y_range],
            bins=hist2d_bins,
            cmap="Blues",
            norm="log"
        )

        # --------------------------
        # Add titles and labels
        # --------------------------

        if use_compared_titles:
            if jj == 0:
                panel_label = "original"
            elif jj - 1 < len(diff_labels):
                panel_label = diff_labels[jj - 1]
            else:
                panel_label = f"run {jj}"
        else:
            panel_label = str(labels[jj]) if jj < len(labels) else f"sample_{jj}"
        ax.set_title(panel_label, pad=8, fontsize=10)

        # Axis labels
        ax.set_xlabel(r"$\log(1/\Delta R)$")
        ax.set_ylabel(r"$\log(k_t)$")

    for offset, (label, reason) in enumerate(unavailable_notes):
        ax = used_axs[Nin + offset]
        ax.set_title(label)
        ax.text(
            0.5,
            0.5,
            f"Plot unavailable\n{reason}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            wrap=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in flat_axs[Nplots:]:
      ax.set_visible(False)

    # --------------------------
    # Add colorbar (shared)
    # --------------------------
    if last_hist is not None:
      cbar = fig.colorbar(last_hist[3], ax=used_axs.tolist(), fraction=0.022, pad=0.055)
      cbar.set_label("Density (log scale)")

    fig.suptitle("Lund Plane Distribution", fontsize=14, y=0.985)
    caption_lines = _plot_note_from_common(common_items, unavailable_notes)
    bottom_margin = 0.13 if caption_lines else 0.09
    fig.subplots_adjust(left=0.08, right=0.88, bottom=bottom_margin, top=0.90, wspace=0.30, hspace=0.55)
    if caption_lines:
      fig.text(
        0.08,
        0.025,
        _wrapped_note(caption_lines, width=150),
        ha="left",
        va="bottom",
        fontsize=8,
      )

    # Save
    name = "lund"
    fig.savefig(os.path.join(out_dir, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Plotting lund-plot to {out_dir}/{name}.pdf")

def resolve_hist2d_shape(nplots, hist2d_shape=None):
  if hist2d_shape is not None:
    nrows, ncols = hist2d_shape
    if nrows < 1 or ncols < 1:
      raise ValueError("--hist2d-shape values must be positive")
    if nrows * ncols < nplots:
      raise ValueError(
          f"--hist2d-shape {nrows} {ncols} has only {nrows * ncols} slots for {nplots} plots"
      )
    return nrows, ncols

  if nplots <= 3:
    return 1, nplots
  if nplots == 4:
    return 2, 2

  ncols = math.ceil(math.sqrt(nplots))
  nrows = math.ceil(nplots / ncols)
  return nrows, ncols

def _default_hist1d_ranges(inputs, display_order, quantiles=(0.5, 99.5), margin_fraction=0.05):
    ranges = []
    reference = inputs[0]
    for ii in range(reference.shape[1]):
        feature_idx = display_order[ii] if ii < len(display_order) else ii
        values = reference[:, feature_idx]
        values = values[np.isfinite(values)]
        if values.size == 0:
            ranges.append([-1.0, 1.0])
            continue

        vmin, vmax = np.percentile(values, quantiles)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = np.min(values)
            vmax = np.max(values)

        margin = margin_fraction * (vmax - vmin)
        if margin == 0:
            margin = 1.0
        ranges.append([vmin - margin, vmax + margin])
    return ranges

def _symmetric_limit(values, fallback=1.0):
    finite = []
    for value in values:
        arr = np.asarray(value)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            finite.append(np.abs(arr))
    if not finite:
        return fallback
    merged = np.concatenate(finite)
    if merged.size == 0:
        return fallback
    vmax = np.percentile(merged, 99.0)
    vmax = max(vmax, np.max(merged) if vmax == 0 else vmax)
    return vmax if vmax > 0 else fallback

def plot_combined_1dhist_ratio_diff(
    inputs,
    labels=None,
    out_dir="./Plots/",
    hist1d_ranges=None,
    hist1d_bins=30,
    logy=False,
    out_name=None,
    unavailable_notes=None,
    min_reference_count=5,
    max_abs_diff=1.0,
):
    if len(inputs) <= 1:
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if labels is None:
        labels = [f"sample_{i}" for i in range(len(inputs))]
    diff_labels, common_items = _caption_comparison(labels, first_run_idx=1)
    comparison_labels = diff_labels if len(diff_labels) > 0 else labels[1:]

    Ndim = inputs[0].shape[1]
    display_order = [0, 1] if Ndim >= 2 else list(range(Ndim))

    if hist1d_ranges is None:
        hist1d_ranges = _default_hist1d_ranges(inputs, display_order)

    fig, axs = plt.subplots(Ndim, 1, figsize=(8.0, 8.0))
    if Ndim == 1:
        axs = [axs]

    axis_titles = [r"$\log(k_t)$", r"$\log(1/\Delta R)$"]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    for ii in range(Ndim):
        feature_idx = display_order[ii] if ii < len(display_order) else ii
        reference_counts, bin_edges = np.histogram(
            inputs[0][:, feature_idx],
            bins=hist1d_bins,
            range=hist1d_ranges[ii],
            density=False,
        )
        reference_total = np.sum(reference_counts)
        reference_density = reference_counts / reference_total if reference_total > 0 else reference_counts
        populated_reference = reference_counts >= max(1, int(min_reference_count))
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ratio_values = []

        for jj, arr in enumerate(inputs[1:], start=1):
            comparison_counts, _ = np.histogram(
                arr[:, feature_idx],
                bins=hist1d_bins,
                range=hist1d_ranges[ii],
                density=False,
            )
            comparison_total = np.sum(comparison_counts)
            comparison_density = comparison_counts / comparison_total if comparison_total > 0 else comparison_counts
            ratio = np.full_like(reference_density, np.nan, dtype=float)
            np.divide(
                reference_density - comparison_density,
                reference_density,
                out=ratio,
                where=populated_reference & (reference_density > 0.0),
            )
            ratio_values.append(ratio)
            label_idx = jj - 1
            axs[ii].step(
                centers,
                ratio,
                where="mid",
                linestyle=linestyles[label_idx % len(linestyles)],
                label=comparison_labels[label_idx] if label_idx < len(comparison_labels) else f"sample_{jj}",
            )

        axs[ii].axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
        axs[ii].set_title(axis_titles[ii] if ii < len(axis_titles) else f"feature_{ii}")
        axs[ii].set_xlabel("value")
        axs[ii].set_ylabel("Fractional diff" + (" (symlog)" if logy else ""))

        ymax = _symmetric_limit(ratio_values, fallback=1.0) * 1.15
        if max_abs_diff is not None and max_abs_diff > 0:
            ymax = min(ymax, float(max_abs_diff))
        if logy:
            axs[ii].set_yscale("symlog", linthresh=1.0)
            axs[ii].set_ylim(-ymax, ymax)
        else:
            axs[ii].set_ylim(-ymax, ymax)

    handles, legend_labels = axs[0].get_legend_handles_labels()
    note_lines = _plot_note_from_common(common_items, unavailable_notes)
    note_lines.append(
        f"Masked original bins with < {max(1, int(min_reference_count))} entries; y clipped at +/- {max_abs_diff:g}"
        if max_abs_diff is not None and max_abs_diff > 0
        else f"Masked original bins with < {max(1, int(min_reference_count))} entries"
    )
    bottom_edge = 0.20 if note_lines else 0.08
    fig.tight_layout(rect=[0.0, bottom_edge, 1.0, 1.0])
    if len(handles) > 0:
        for ax in axs:
            ax.legend(handles, legend_labels, loc="best", fontsize=8, framealpha=0.90)
    if note_lines:
        fig.text(
            0.02,
            0.02,
            _wrapped_note(note_lines, width=145),
            ha="left",
            va="bottom",
            fontsize=8,
        )

    if out_name is None:
        out_name = "hist1d_ratio_diff"

    fig.savefig(os.path.join(out_dir, out_name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, out_name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Plotting projection diff to {out_dir}/{out_name}.pdf")

def plot_combined_1dhist(
    inputs,
    labels=None,
    out_dir="./Plots/",
    hist1d_ranges=None,
    hist1d_bins=30,
    logy=False,
    out_name=None,
    logy_floor_mode="clamped",
    unavailable_notes=None,
):
    """
    Plot overlaid 1D histograms for Lund features.

    Expected feature order:
      inputs[:, 0] = log(kt)
      inputs[:, 1] = log(1/deltaR)

    The top panel is log(kt), and the bottom panel is log(1/deltaR).
    This matches the plotting convention used in plot_ublund.py.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if labels is None:
        labels = [f"sample_{i}" for i in range(len(inputs))]
    diff_labels, common_items = _caption_comparison(labels, first_run_idx=1)
    plot_labels = ["original"]
    plot_labels.extend(diff_labels)

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    Ndim = inputs[0].shape[1]
    display_order = [0, 1] if Ndim >= 2 else list(range(Ndim))

    if hist1d_ranges is None:
        hist1d_ranges = _default_hist1d_ranges(inputs, display_order)

    fig, axs = plt.subplots(Ndim, 1, figsize=(8.0, 8.0))
    if Ndim == 1:
        axs = [axs]

    axis_titles = [r"$\log(k_t)$", r"$\log(1/\Delta R)$"]

    for ii in range(Ndim):
        # For ktdr plots, keep the top panel aligned with the Lund y-axis:
        # log(kt) first, then log(1/deltaR).
        feature_idx = display_order[ii] if ii < len(display_order) else ii

        all_hist_counts = []
        positive_hist_counts = []

        for jj, arr in enumerate(inputs):
            values = arr[:, feature_idx]

            hist_counts, _ = np.histogram(
                values,
                bins=hist1d_bins,
                range=hist1d_ranges[ii],
                density=True,
            )

            all_hist_counts.append(hist_counts)
            positive_hist_counts.extend(hist_counts[hist_counts > 0.0])

            axs[ii].hist(
                values,
                bins=hist1d_bins,
                range=hist1d_ranges[ii],
                histtype="step",
                density=True,
                linestyle=linestyles[jj % len(linestyles)],
                label=plot_labels[jj] if jj < len(plot_labels) else f"sample_{jj}",
            )

        axs[ii].set_title(axis_titles[ii] if ii < len(axis_titles) else f"feature_{ii}")
        axs[ii].set_xlabel("value")
        axs[ii].set_ylabel("Density (log scale)" if logy else "Density")

        ymax = max([np.max(h) for h in all_hist_counts if h.size > 0], default=1.0)

        if logy:
            axs[ii].set_yscale("log")
            if len(positive_hist_counts) > 0:
                min_positive = min(positive_hist_counts)
                max_positive = max(positive_hist_counts)

                if logy_floor_mode == "tail":
                    y_min = min_positive / 3.0
                else:
                    y_min = max(min_positive / 3.0, 1e-4)

                y_max = max_positive * 1.5
                if y_max <= y_min:
                    y_max = y_min * 10.0

                axs[ii].set_ylim(y_min, y_max)
            else:
                axs[ii].set_ylim(1e-4, 1.0)
        else:
            axs[ii].set_ylim(0.0, ymax * 1.15 if ymax > 0 else 1.0)

    handles, legend_labels = axs[0].get_legend_handles_labels()
    note_lines = _plot_note_from_common(common_items, unavailable_notes)
    has_caption = bool(note_lines)
    bottom_edge = 0.20 if has_caption else 0.08
    fig.tight_layout(rect=[0.0, bottom_edge, 1.0, 1.0])
    if len(handles) > 0:
        for ax in axs:
            ax.legend(
                handles,
                legend_labels,
                loc="best",
                fontsize=8,
                framealpha=0.90,
            )
    if note_lines:
        fig.text(
            0.02,
            0.02,
            _wrapped_note(note_lines, width=145),
            ha="left",
            va="bottom",
            fontsize=8,
        )

    if out_name is None:
        out_name = "hist1d_logy" if logy else "hist1d"

    fig.savefig(os.path.join(out_dir, out_name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, out_name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Plotting hist to {out_dir}/{out_name}.pdf")

def plot_lund_ratio_diff(
    inputs,
    labels=["original","generated","predicted"],
    out_dir="./Plots/",
    hist2d_xrange=None,
    hist2d_yrange=None,
    hist2d_bins=(20, 20),
    hist2d_shape=None,
    hist2d_layout=None,
    unavailable_notes=None,
    min_reference_count=5,
    max_abs_diff=1.0,
):
  if len(inputs) <= 1:
    return

  Ndim = inputs[0].shape[1]
  if Ndim < 2:
    return

  unavailable_notes = unavailable_notes or []
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  Ndiff = len(inputs) - 1
  Nplots = Ndiff + len(unavailable_notes)
  if Nplots == 0:
    return

  diff_labels, common_items = _caption_comparison(labels, first_run_idx=1)
  comparison_labels = diff_labels if len(diff_labels) > 0 else labels[1:]

  mins = np.zeros(Ndim)
  maxs = np.zeros(Ndim)
  for ii in range(Ndim):
    for jj in range(len(inputs)):
      mins[ii] = min(mins[ii], np.min(inputs[jj][:, ii]))
      maxs[ii] = max(maxs[ii], np.max(inputs[jj][:, ii]))

  mins[1] = -3
  maxs[0] = 8
  x_range = hist2d_xrange if hist2d_xrange is not None else [mins[1], maxs[1]]
  y_range = hist2d_yrange if hist2d_yrange is not None else [mins[0], maxs[0]]

  reference_counts, x_edges, y_edges = np.histogram2d(
      inputs[0][:, 1],
      inputs[0][:, 0],
      range=[x_range, y_range],
      bins=hist2d_bins,
      density=False,
  )
  reference_total = np.sum(reference_counts)
  reference_density = reference_counts / reference_total if reference_total > 0 else reference_counts
  populated_reference = reference_counts >= max(1, int(min_reference_count))

  ratio_maps = []
  for arr in inputs[1:]:
    comparison_counts, _, _ = np.histogram2d(
        arr[:, 1],
        arr[:, 0],
        range=[x_range, y_range],
        bins=hist2d_bins,
        density=False,
    )
    comparison_total = np.sum(comparison_counts)
    comparison_density = comparison_counts / comparison_total if comparison_total > 0 else comparison_counts
    ratio_map = np.full_like(reference_density, np.nan, dtype=float)
    np.divide(
        reference_density - comparison_density,
        reference_density,
        out=ratio_map,
        where=populated_reference & (reference_density > 0.0),
    )
    ratio_maps.append(ratio_map)

  if hist2d_shape is None:
    hist2d_shape = hist2d_layout
  nrows, ncols = resolve_hist2d_shape(Nplots, hist2d_shape)
  fig, axs = plt.subplots(
      nrows,
      ncols,
      figsize=(5.0 * ncols + 1.0, 4.4 * nrows + 0.6),
      squeeze=False,
  )
  flat_axs = axs.ravel()
  used_axs = flat_axs[:Nplots]

  vmax = _symmetric_limit(ratio_maps, fallback=1.0)
  if max_abs_diff is not None and max_abs_diff > 0:
    vmax = min(vmax, float(max_abs_diff))
  norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
  cmap = plt.get_cmap("coolwarm").copy()
  cmap.set_bad(color="#d9d9d9")
  last_image = None

  for jj, ratio_map in enumerate(ratio_maps):
    ax = used_axs[jj]
    last_image = ax.imshow(
        ratio_map.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(comparison_labels[jj] if jj < len(comparison_labels) else f"sample_{jj + 1}", pad=8, fontsize=10)
    ax.set_xlabel(r"$\log(1/\Delta R)$")
    ax.set_ylabel(r"$\log(k_t)$")

  for offset, (label, reason) in enumerate(unavailable_notes):
    ax = used_axs[Ndiff + offset]
    ax.set_title(label)
    ax.text(
        0.5,
        0.5,
        f"Plot unavailable\n{reason}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])

  for ax in flat_axs[Nplots:]:
    ax.set_visible(False)

  if last_image is not None:
    cbar = fig.colorbar(last_image, ax=used_axs.tolist(), fraction=0.022, pad=0.055, extend="both")
    cbar.set_label(r"$(original - generated) / original$")

  fig.suptitle("Lund Plane Fractional Difference", fontsize=14, y=0.985)
  note_lines = _plot_note_from_common(common_items, unavailable_notes)
  note_lines.append(
      f"Masked original bins with < {max(1, int(min_reference_count))} entries; color clipped at +/- {vmax:.3g}"
  )
  bottom_margin = 0.13 if note_lines else 0.09
  fig.subplots_adjust(left=0.08, right=0.88, bottom=bottom_margin, top=0.90, wspace=0.30, hspace=0.55)
  if note_lines:
    fig.text(
        0.08,
        0.025,
        _wrapped_note(note_lines, width=150),
        ha="left",
        va="bottom",
        fontsize=8,
    )

  name = "lund_ratio_diff"
  fig.savefig(os.path.join(out_dir, name + ".png"), bbox_inches="tight")
  fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight")
  plt.close(fig)
  print(f"Plotting lund-plane diff to {out_dir}/{name}.pdf")

# ---------------------------------------------------------------------
# Main validation function which runs all the plots
# ---------------------------------------------------------------------
def validate_unbinned_models(models, test_loader, args, labels=None, make_projection=False, unavailable_model_reasons=None,):
  if args.plot_max_batches is not None and args.plot_max_batches <= 0:
    raise ValueError("--plot-max-batches must be a positive integer")

  if labels is None:
    labels = ["original"]
    if len(models) == 1:
      labels.append("generated")
    else:
      labels.extend([f"generated_{ii}" for ii in range(len(models))])

  with torch.no_grad():
      original_chunks = []
      generated_chunks = [[] for _ in models]
      active_models = [True for _ in models]
      unavailable_reasons = [None for _ in models]

      device = "cpu"

      # ---------------------------------------------------------------------
      # Sanity checks
      # ---------------------------------------------------------------------
      for imodel, model in enumerate(models):
        model.to(device)
        model.eval()
        forced_reason = None
        if unavailable_model_reasons is not None and imodel < len(unavailable_model_reasons):
          forced_reason = unavailable_model_reasons[imodel]
        if forced_reason:
          reason = forced_reason
          print(f"Generated plot unavailable for model {imodel}: {reason}.", flush=True)
          active_models[imodel] = False
          unavailable_reasons[imodel] = reason
        elif helpers.model_has_nonfinite_parameters(model):
          reason = "non-finite model parameters"
          print(f"Generated plot unavailable for model {imodel}: {reason}.", flush=True)
          active_models[imodel] = False
          unavailable_reasons[imodel] = reason

      # ---------------------------------------------------------------------
      # Loop over epochs and generate
      # ---------------------------------------------------------------------
      printed_example = False
      starttime=time.time()
      Nimages=0
      for batch, X in enumerate(test_loader):
        if args.plot_max_batches is not None and batch >= args.plot_max_batches:
          break
        if batch>1000: break #FIXME

        #X = X.to(device)
        original_chunks.append(X)

        for imodel, model in enumerate(models):
          if not active_models[imodel]:
            continue

          try:
              generated_seq = model.generate(out_dimensions=X.shape)
          except RuntimeError as err:
            reason = f"generation failed: {err}"
            print(f"Generated plot unavailable for model {imodel}: {reason}", flush=True)
            active_models[imodel] = False
            generated_chunks[imodel] = []
            unavailable_reasons[imodel] = reason
            continue

          if not torch.isfinite(generated_seq).all():
            reason = "generated sequence contains nan/inf"
            print(f"Generated plot unavailable for model {imodel}: {reason}.", flush=True)
            active_models[imodel] = False
            generated_chunks[imodel] = []
            unavailable_reasons[imodel] = reason
            continue

          if args.mixed_loss:
            generated_seq[:, :, -1] = torch.sigmoid(generated_seq[:, :, -1])
            generated_seq[:, 0, -1] = X[:, 0, -1]

          if not printed_example:
            print("Input example")
            print(X[0])
            print("Generate example")
            starttime_single=time.time()
            print(generated_seq[0])
            print("Took %.2e ms to generate 1 image"%((time.time()-starttime_single)*1000), flush=True)
            printed_example = True

          generated_chunks[imodel].append(generated_seq)

      print("Took %.2f min to generate %i images"%((time.time()-starttime)/60,len(test_loader.dataset)), flush=True) 

      if len(original_chunks) == 0:
        raise ValueError("No validation batches were plotted")

      
      original = torch.cat(original_chunks)
      generated_list = [
          torch.cat(chunks)
          for chunks in generated_chunks
          if len(chunks) > 0
      ]
      active_labels = [labels[0]]
      active_labels.extend(
          labels[imodel + 1]
          for imodel, chunks in enumerate(generated_chunks)
          if len(chunks) > 0 and imodel + 1 < len(labels)
      )
      unavailable_notes = [
          (
              labels[imodel + 1] if imodel + 1 < len(labels) else f"generated_{imodel}",
              reason,
          )
          for imodel, reason in enumerate(unavailable_reasons)
          if reason is not None
      ]

      flat_original = original.flatten(0, 1).cpu().numpy()
      flat_generated_list = [g.flatten(0, 1).cpu().numpy() for g in generated_list]
      plot_inputs = [flat_original] + flat_generated_list

      # ---------------------------------------------------------------------
      # Make the plots
      # ---------------------------------------------------------------------
      if make_projection:
        projection_plot(
            plot_inputs,
            labels=active_labels,
            out_dir=args.plot_dir,
            name="projection",
            unavailable_notes=unavailable_notes,
        )

      hist1d_ranges = None
      if args.hist1d_ranges is not None:
        if len(args.hist1d_ranges) != 4:
          raise ValueError("--hist1d-ranges should be: kt_min kt_max dr_min dr_max")
        hist1d_ranges = [
          [args.hist1d_ranges[0], args.hist1d_ranges[1]],
          [args.hist1d_ranges[2], args.hist1d_ranges[3]],
        ]

      plot_combined_1dhist(
          plot_inputs,
          labels=active_labels,
          out_dir=args.plot_dir,
          hist1d_ranges=hist1d_ranges,
          hist1d_bins=args.hist1d_bins,
          logy=False,
          out_name="hist1d",
          unavailable_notes=unavailable_notes,
      )

      plot_combined_1dhist(
          plot_inputs,
          labels=active_labels,
          out_dir=args.plot_dir,
          hist1d_ranges=hist1d_ranges,
          hist1d_bins=args.hist1d_bins,
          logy=True,
          out_name="hist1d_logy",
          unavailable_notes=unavailable_notes,
      )

      #Make plot
      if args.hist_ratio_diff and len(plot_inputs) > 1:
        plot_combined_1dhist_ratio_diff(
            plot_inputs,
            labels=active_labels,
            out_dir=args.plot_dir,
            hist1d_ranges=hist1d_ranges,
            hist1d_bins=args.hist1d_bins,
            logy=False,
            out_name="hist1d_ratio_diff",
            unavailable_notes=unavailable_notes,
            min_reference_count=args.hist_ratio_min_count,
            max_abs_diff=args.hist_ratio_vmax,
        )

      if args.input_format == "ktdr":
        lund_inputs = plot_inputs
      else:
        starttime=time.time()
        lund_original = helpers.make_lundplane(original)
        lund_inputs = [lund_original.reshape(-1, lund_original.shape[-1])]
        print("Took %.2f min to make the lund-plane"%((time.time()-starttime)/60), flush=True)
        for generated in generated_list:
          lund_generated = helpers.make_lundplane(generated)
          lund_inputs.append(lund_generated.reshape(-1, lund_generated.shape[-1]))


        if make_projection:
            projection_plot(
                lund_inputs,
                labels=active_labels,
                out_dir=args.plot_dir,
                name="projection_lund",
                unavailable_notes=unavailable_notes,
            )

      lund_plot(
          lund_inputs,
          labels=active_labels,
          out_dir=args.plot_dir,
          hist2d_xrange=args.hist2d_xrange,
          hist2d_yrange=args.hist2d_yrange,
          hist2d_bins=args.hist2d_bins,
          hist2d_shape=args.hist2d_shape,
          unavailable_notes=unavailable_notes,
      )

      if args.hist_ratio_diff and len(lund_inputs) > 1:
        plot_lund_ratio_diff(
            lund_inputs,
            labels=active_labels,
            out_dir=args.plot_dir,
            hist2d_xrange=args.hist2d_xrange,
            hist2d_yrange=args.hist2d_yrange,
            hist2d_bins=args.hist2d_bins,
            hist2d_shape=args.hist2d_shape,
            unavailable_notes=unavailable_notes,
            min_reference_count=args.hist_ratio_min_count,
            max_abs_diff=args.hist_ratio_vmax,
        )
