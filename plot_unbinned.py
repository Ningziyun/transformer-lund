from helpers_unbinned import *

def validate(models, test_loader, input_shape, args):
  with torch.no_grad():
      original = torch.empty([0, input_shape[-2], input_shape[-1]])

      # Store generated samples from each model separately.
      generated_list = [
          torch.empty([0, input_shape[-2], input_shape[-1]])
          for _ in models
      ]

      device = "cpu"

      for batch, X in enumerate(test_loader):
        '''
        if batch >= 5:
          break
        '''
        X = X.to(device)
        start = X[:, 0, :].unsqueeze(1)

        # Original data is shared by all model comparisons.
        original = torch.cat([original, X])

        for imodel, model in enumerate(models):
          model.to(device)
          model.eval()

          generated_seq = model.generate(start, steps=X.shape[1] - 1)

          if args.mixed_loss:
            generated_seq[:, :, -1] = sigmoid(generated_seq[:, :, -1])
            generated_seq[:, 0, -1] = X[:, 0, -1]

          if batch == 0 and imodel == 0:
            print("Input example")
            print(X[0])
            print("Generate example")
            print(generated_seq[0])

          generated_list[imodel] = torch.cat(
              [generated_list[imodel], generated_seq]
          )

      hist1d_ranges = None
      if args.hist1d_ranges is not None:
        if len(args.hist1d_ranges) != 4:
          raise ValueError("--hist1d-ranges should be: kt_min kt_max dr_min dr_max")
        hist1d_ranges = [
          [args.hist1d_ranges[0], args.hist1d_ranges[1]],
          [args.hist1d_ranges[2], args.hist1d_ranges[3]],
        ]

      flat_original = original.flatten(0, 1).numpy()
      flat_generated_list = [
          g.flatten(0, 1).numpy()
          for g in generated_list
      ]

      labels = ["original"]
      for path in args.model_path:
        labels.append(os.path.basename(path).replace(".pt", ""))

      plot_inputs = [flat_original] + flat_generated_list

      plot_combined_1dhist(
          plot_inputs,
          labels=labels,
          out_dir=args.log_dir,
          hist1d_ranges=hist1d_ranges,
          hist1d_bins=args.hist1d_bins,
          logy=False,
          out_name="hist1d",
      )

      plot_combined_1dhist(
          plot_inputs,
          labels=labels,
          out_dir=args.log_dir,
          hist1d_ranges=hist1d_ranges,
          hist1d_bins=args.hist1d_bins,
          logy=True,
          out_name="hist1d_logy",
      )

      if args.input_format == "ktdr":
        lund_plot(
            plot_inputs,
            labels=labels,
            outdir=args.log_dir,
            hist2d_xrange=args.hist2d_xrange,
            hist2d_yrange=args.hist2d_yrange,
            hist2d_bins=args.hist2d_bins,
        )
      else:
        lund_inputs = [make_lundplane(original.numpy()).reshape(-1, input_shape[-1])]

        for g in generated_list:
          lund_g = make_lundplane(g.numpy())
          lund_inputs.append(lund_g.reshape(-1, lund_g.shape[-1]))

        lund_plot(
            lund_inputs,
            labels=labels,
            outdir=args.log_dir,
            hist2d_xrange=args.hist2d_xrange,
            hist2d_yrange=args.hist2d_yrange,
            hist2d_bins=args.hist2d_bins,
        )


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

    validate(models, test_loader, X_example.shape, args)


if __name__ == "__main__":
    main()