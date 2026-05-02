from helpers_unbinned import *

def main():
    args = parse_input()
    device = "cpu"

    if args.log_dir == "models/test" and len(args.model_path) > 0:
        # For multiple models, use the directory of the first model as the default output directory.
        args.log_dir = os.path.dirname(os.path.abspath(args.model_path[0]))

    train_loader, test_loader = get_loaders(
        args.input_format,
        train_file=args.train_file,
        val_file=args.val_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    X_example = next(iter(train_loader))

    # Load all models passed from --model-path.
    models = []
    for model_path in args.model_path:
        print(f"Loading model: {model_path}")
        model = load_model(model_path)
        model.to(device)
        model.eval()
        models.append(model)

    labels = ["original"]
    for path in args.model_path:
        labels.append(os.path.basename(path).replace(".pt", ""))

    validate_unbinned_models(
        models,
        test_loader,
        X_example.shape,
        args,
        labels=labels,
        make_projection=False,
    )


if __name__ == "__main__":
    main()
