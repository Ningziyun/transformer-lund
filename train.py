import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model import JetTransformer

from tqdm import tqdm
from helpers_train import *

torch.multiprocessing.set_sharing_strategy("file_system")


if __name__ == "__main__":
    args = parse_input()
    save_arguments(args)
    print(f"Logging to {args.log_dir}")
    set_seeds(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    num_features = 3
    num_bins = tuple(args.num_bins)

    print(f"Using bins: {num_bins}")
    print(f"{'Not' if not args.reverse else ''} reversing pt order")

    # load and preprocess data
    print(f"Loading training set")
    train_loader = load_data(
        path=args.data_path,
        n_events=args.num_events,
        num_features=num_features,
        num_bins=num_bins,
        num_const=args.num_const,
        reverse=args.reverse,
        start_token=args.start_token,
        end_token=args.end_token,
        limit_const=args.limit_const,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Loading validation set")
    val_loader = load_data(
        path=args.data_path.replace("train", "val"),
        n_events=100000,
        num_features=num_features,
        num_bins=num_bins,
        num_const=args.num_const,
        reverse=args.reverse,
        start_token=args.start_token,
        end_token=args.end_token,
        limit_const=args.limit_const,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # construct model
    if args.contin:
        model = load_model(log_dir=args.model_path)
        print("Loaded model")
    else:
        model = JetTransformer(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_features=num_features,
            num_bins=num_bins,
            dropout=args.dropout,
            output=args.output,
            tanh=args.tanh,
            end_token=args.end_token,
        )
    model.to(device)

    # construct optimizer and auto-caster
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_cos_scheduler(args.num_epochs, len(train_loader), opt)
    scaler = torch.cuda.amp.GradScaler()

    if args.contin:
        load_opt_states(opt, scheduler, scaler, args.log_dir)
        print("Loaded optimizer")

    logger = SummaryWriter(args.log_dir)
    global_step = args.global_step
    loss_list = []
    perplexity_list = []
    for epoch in range(args.num_epochs):
        model.train()

        for x, padding_mask, true_bin in tqdm(
            train_loader, total=len(train_loader), desc=f"Training Epoch {epoch + 1}"
        ):
            opt.zero_grad()
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            true_bin = true_bin.to(device)

            with torch.cuda.amp.autocast():
                logits = model(x, padding_mask)
                loss = model.loss(logits, true_bin)
                with torch.no_grad():
                    perplexity = model.probability(
                        logits,
                        padding_mask,
                        true_bin,
                        perplexity=True,
                        logarithmic=False,
                        topk=False
                    )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            loss_list.append(loss.cpu().detach().numpy())
            perplexity_list.append(perplexity.mean().cpu().detach().numpy())

            if (global_step + 1) % args.logging_steps == 0:
                logger.add_scalar("Train/Loss", np.mean(loss_list), global_step)
                logger.add_scalar(
                    "Train/Perplexity", np.mean(perplexity_list), global_step
                )
                logger.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
                loss_list = []
                perplexity_list = []

            if (args.checkpoint_steps != 0) and (
                (global_step + 1) % args.checkpoint_steps == 0
            ):
                save_model(model, args.log_dir, f"checkpoint_{global_step + 1}")

            global_step += 1

        model.eval()
        with torch.no_grad():
            val_loss = []
            val_perplexity = []
            for x, padding_mask, true_bin in tqdm(
                val_loader, total=len(val_loader), desc=f"Validation Epoch {epoch + 1}"
            ):
                x = x.to(device)
                padding_mask = padding_mask.to(device)
                true_bin = true_bin.to(device)

                logits = model(
                    x,
                    padding_mask,
                )
                loss = model.loss(logits, true_bin)
                perplexity = model.probability(
                    logits, padding_mask, true_bin, perplexity=True, logarithmic=False
                )
                val_loss.append(loss.cpu().detach().numpy())
                val_perplexity.append(perplexity.mean().cpu().detach().numpy())

            logger.add_scalar("Val/Loss", np.mean(val_loss), global_step)
            logger.add_scalar("Val/Perplexity", np.mean(val_perplexity), global_step)

        save_model(model, args.log_dir, "last")
        save_opt_states(opt, scheduler, scaler, args.log_dir)
