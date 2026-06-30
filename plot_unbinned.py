from helpers_unbinned import *

def _format_scalar(value):
    as_float = _as_float(value, None)
    if as_float is None:
        return str(value)
    return f"{as_float:g}"

def _display_epoch_from_meta(meta, key):
    display = _as_int(meta.get(f"{key}_display", None), None)
    if display is not None:
        return display
    index = _as_int(meta.get(key, None), None)
    if index is None:
        return None
    return index + 1

def _artifact_label_from_name(name):
    if name is None:
        return None
    base = os.path.splitext(os.path.basename(str(name)))[0]
    if base == "best":
        return "best"
    match = re.search(r"epoch[_-](\d+)", base)
    if match is not None:
        return f"ep {int(match.group(1)) + 1}"
    return base if base else None

def load_unbinned_model_for_plot(model_path, input_dim, device="cpu"):
    obj = load_model(model_path, map_location="cpu")
    metadata = {}

    if isinstance(obj, dict) and "model_state_dict" in obj:
        ckpt_args = obj.get("args", {})
        metadata.update(ckpt_args)
        for key in (
            "artifact_type",
            "save_mode",
            "epoch",
            "epoch_display",
            "loss",
            "best_epoch",
            "best_epoch_display",
            "best_loss",
            "model_mode",
            "scheduler",
            "optimizer",
            "weight_decay",
            "grad_clip",
            "current_lr",
            "next_lr",
        ):
            if key in obj:
                metadata[key] = obj[key]
        metadata["resolved_model_mode"] = obj.get("model_mode", resolved_model_mode(ckpt_args))
        model = build_unbinned_model(input_dim, ckpt_args)
        model.load_state_dict(obj["model_state_dict"])
    else:
        model = obj

    model.to(device)
    model.eval()
    return model, metadata

def build_run_caption(meta, fallback=None):
    model_mode = meta.get("resolved_model_mode", meta.get("model_mode", "model"))
    batch_size = meta.get("batch_size", meta.get("batch-size", "?"))
    lr = meta.get("lr", "?")
    epochs = meta.get("epochs", meta.get("total_epochs", None))
    scheduler = meta.get("scheduler", "none")
    input_format = meta.get("input_format", meta.get("input-format", None))
    n_mix = meta.get("n_mix", meta.get("n-mix", None))
    best_epoch_display = _display_epoch_from_meta(meta, "best_epoch")
    epoch_display = _display_epoch_from_meta(meta, "epoch")

    artifact_label = None
    fallback_label = _artifact_label_from_name(fallback)
    if fallback_label == "best":
        artifact_label = f"best ep {best_epoch_display}" if best_epoch_display is not None else "best"
    elif epoch_display is not None:
        artifact_label = f"ep {epoch_display}"
    else:
        artifact_label = fallback_label

    first_line = [str(model_mode)]
    if artifact_label is not None:
        first_line.append(artifact_label)
    if best_epoch_display is not None and (artifact_label is None or not artifact_label.startswith("best")):
        first_line.append(f"best ep {best_epoch_display}")

    second_line = []
    if input_format is not None:
        second_line.append(str(input_format))
    if n_mix is not None and _as_bool(meta.get("mdn", False)):
        second_line.append(f"{n_mix} Gauss")
    second_line.extend([f"bs {batch_size}", f"lr {_format_scalar(lr)}"])
    if epochs is not None:
        second_line.append(f"tot {epochs}")

    lines = [" ".join(first_line), " ".join(second_line)]
    if scheduler not in (None, "none"):
        sched_label = {
            "cos_damping": "cos damp",
            "cosine": "cos",
            "plateau": "plateau",
        }.get(str(scheduler), str(scheduler))
        sched_line = [f"sched {sched_label}"]
        if scheduler == "cos_damping":
            start_epoch = meta.get("cos_damping_start_epoch", meta.get("cos_start_epoch", None))
            start_display = _epoch_display_from_index(start_epoch)
            final_lr = meta.get("cos_damping_final_lr", meta.get("cos_final_lr", None))
            if start_display is not None:
                sched_line.append(f"start ep {start_display}")
            if final_lr is not None:
                sched_line.append(f"lr_f {_format_scalar(final_lr)}")
        elif scheduler in ("cosine", "plateau") and meta.get("scheduler_min_lr", None) is not None:
            sched_line.append(f"min lr {_format_scalar(meta.get('scheduler_min_lr'))}")
        lines.append(" ".join(sched_line))
    return "\n".join(lines)

def infer_log_dir_from_path(model_path):
    parent = os.path.dirname(os.path.abspath(model_path))
    if os.path.basename(parent) == "checkpoints":
        return os.path.dirname(parent)
    return parent

def parse_arguments_txt(txt_path):
    meta = {}
    if not os.path.exists(txt_path):
        return meta
    with open(txt_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
                continue
            parts = line.split()
            if len(parts) >= 2:
                meta[parts[0].strip()] = " ".join(parts[1:]).strip()
    return meta

def main():
    args = parse_input()
    device = "cpu"

    if len(args.model_path) == 0:
        raise ValueError("Please pass at least one model/checkpoint with --model-path or --checkpoint")

    if args.log_dir == "models/test":
        # For multiple models, use the log directory of the first model/checkpoint as the default output directory.
        args.log_dir = infer_log_dir_from_path(args.model_path[0])
    plot_dir = args.plot_dir or args.log_dir
    args.log_dir = plot_dir

    train_loader, test_loader = get_loaders(
        args.input_format,
        train_file=args.train_file,
        val_file=args.val_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    X_example = next(iter(train_loader))

    # Load all models/checkpoints passed from --model-path/--checkpoint.
    models = []
    labels = ["original"]
    unavailable_model_reasons = []
    run_infos = []
    for model_path in args.model_path:
        print(f"Loading model: {model_path}")
        model, ckpt_meta = load_unbinned_model_for_plot(model_path, X_example.shape[2], device=device)
        models.append(model)
        if model_has_nonfinite_parameters(model):
            unavailable_model_reasons.append("checkpoint contains nan/inf parameters")
        else:
            unavailable_model_reasons.append(None)

        log_dir = infer_log_dir_from_path(model_path)
        txt_meta = parse_arguments_txt(os.path.join(log_dir, "arguments.txt"))
        meta = {**ckpt_meta, **txt_meta}
        fallback = os.path.basename(model_path).replace(".pt", "")
        caption = build_run_caption(meta, fallback=fallback)
        labels.append(caption)
        run_infos.append(
            {
                "checkpoint_path": model_path,
                "log_dir": log_dir,
                "caption": caption,
                "loss_curves_csv": load_loss_history_csv(os.path.join(log_dir, "loss_history.csv")),
            }
        )

    validate_unbinned_models(
        models,
        test_loader,
        X_example.shape,
        args,
        labels=labels,
        make_projection=False,
        unavailable_model_reasons=unavailable_model_reasons,
    )

    plot_combined_losses(run_infos=run_infos, out_dir=plot_dir)
    print(f"Plots written to: {plot_dir}")


if __name__ == "__main__":
    main()
