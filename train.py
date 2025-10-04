import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os             
import ROOT           

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

    num_features = args.num_features # Origin was 3(Step-3)
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
    # Step-8 Adding Lost Function
    epoch_loss_list = []  # store average training loss per epoch
    # Step-8 Ending
    # Step-11 Adding Lost Funtion
    # --- collect validation-step training loss within this epoch ---
    epoch_val_loss_list = []
    # Step-11 Ending
    global_step = args.global_step
    loss_list = []
    perplexity_list = []
    for epoch in range(args.num_epochs):
        model.train()
        # Step-8 Adding Lost Function
        # --- collect per-step training loss within this epoch ---
        epoch_losses = []  # loss of all steps in this epoch
        # Step-8 Ending
        for x, padding_mask, true_bin in tqdm(
            train_loader, total=len(train_loader), desc=f"Training Epoch {epoch + 1}"
        ):
            opt.zero_grad()
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            true_bin = true_bin.to(device)

            with torch.cuda.amp.autocast():
                logits = model(x, padding_mask)
                '''
                Step-9 Deleted
                loss = model.loss(logits, true_bin)
                '''
                # Step 9 Adding
                loss = model.loss(logits, true_bin, padding_mask=padding_mask)
                # Step 9 Ending
                # Step-8 Adding (collect per-step loss for this epoch)
                # Record current step loss as a Python float to reduce tensor overhead
                epoch_losses.append(loss.item())
                # Step-8 Ending
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
        # Step-8 Adding (compute and record epoch-level average training loss)
        # Compute the mean of all per-step losses collected in this epoch
        avg_train_loss = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else float("nan")
        epoch_loss_list.append(avg_train_loss)
        # Optionally also log an epoch-level curve to TensorBoard
        logger.add_scalar("Train/Epoch_Avg_Loss", avg_train_loss, epoch)
        # Step-8 Ending

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
                '''
                Step-9 Deleted
                loss = model.loss(logits, true_bin)
                '''
                # Step 9 Adding
                loss = model.loss(logits, true_bin, padding_mask=padding_mask)
                # Step 9 Ending
                perplexity = model.probability(
                    logits, padding_mask, true_bin, perplexity=True, logarithmic=False
                )
                val_loss.append(loss.cpu().detach().numpy())
                val_perplexity.append(perplexity.mean().cpu().detach().numpy())

            logger.add_scalar("Val/Loss", np.mean(val_loss), global_step)
            logger.add_scalar("Val/Perplexity", np.mean(val_perplexity), global_step)
            # Step-11 Adding
            avg_val_loss = float(np.mean(val_loss)) if len(val_loss) > 0 else float("nan")
            epoch_val_loss_list.append(avg_val_loss)
            logger.add_scalar("Val/Epoch_Avg_Loss", avg_val_loss, epoch)
            # Step-11 End

        save_model(model, args.log_dir, "last")
        save_opt_states(opt, scheduler, scaler, args.log_dir)
        
    # Step-8 Adding (dump per-epoch average training loss to ROOT)
    # This writes a separate ROOT file and does NOT interfere with any other ROOT outputs.
    out_path = os.path.join(args.log_dir, "epoch_losses.root")
    f = ROOT.TFile(out_path, "RECREATE")
    tree = ROOT.TTree("loss_tree", "Per-epoch average training loss")

    # Use a 1-element numpy array as a C-like buffer for the branch
    loss_val = np.zeros(1, dtype=np.float32)
    tree.Branch("train_loss", loss_val, "train_loss/F")

    # One entry per epoch
    for l in epoch_loss_list:
        loss_val[0] = l
        tree.Fill()

    tree.Write()
    f.Close()
    print(f"[Done] Saved per-epoch losses to {out_path}")
    # Step-8 Ending
    # Step-11 Adding — dump per-epoch average validation loss to ROOT
    # This writes a separate ROOT file for validation and mirrors the training-loss output.
    out_path_val = os.path.join(args.log_dir, "epoch_losses_val.root")
    f_val = ROOT.TFile(out_path_val, "RECREATE")
    tree_val = ROOT.TTree("val_loss_tree", "Per-epoch average validation loss")

    # Use a 1-element numpy array as a C-like buffer for the branch
    val_loss_val = np.zeros(1, dtype=np.float32)
    tree_val.Branch("val_loss", val_loss_val, "val_loss/F")

    # One entry per epoch
    for l in epoch_val_loss_list:
        val_loss_val[0] = l
        tree_val.Fill()

    tree_val.Write()
    f_val.Close()
    print(f"[Done] Saved per-epoch validation losses to {out_path_val}")
    # Step-11 Ending

