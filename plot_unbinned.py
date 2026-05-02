from helpers_unbinned import *

def main():
    args = parse_input()
    device = "cpu"

    if len(args.model_path) == 0:
        raise ValueError("Please pass at least one model/checkpoint with --model-path or --checkpoint")

    if args.log_dir == "models/test":
        # For multiple models, use the log directory of the first model/checkpoint as the default output directory.
        args.log_dir = infer_log_dir_from_path(args.model_path[0])

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
    run_infos = []
    for model_path in args.model_path:
        print(f"Loading model: {model_path}")
        model, ckpt_meta = load_unbinned_model_for_plot(model_path, X_example.shape[2], device=device)
        models.append(model)

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
    )

    plot_combined_losses(run_infos=run_infos, out_dir=args.log_dir)
    print(f"Plots written to: {args.log_dir}")


if __name__ == "__main__":
    main()
