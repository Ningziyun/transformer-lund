import os,sys
import time
import numpy as np
import h5py
import math
import csv
import re
import matplotlib.pyplot as plt

import argparse

import torch
from torchinfo import summary
from torch.utils.data import DataLoader
import torch.nn.functional as F

import models_generative

import helpers

# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
class ktdr_dataset(torch.utils.data.Dataset):
  def __init__(self, file_path, NConstituents=20, add_stop=False, add_mask=False, standardize=False):
    super(ktdr_dataset, self).__init__()
    self.data=torch.tensor([])
    self.add_stop=add_stop
    self.add_mask=add_mask

    f = h5py.File(file_path,'r')
    drs=f["lundplane"]["dr"]
    kts=f["lundplane"]["kt"]
    self.DR=drs[:,:NConstituents]
    self.kt=kts[:,:NConstituents]

    #Check if next is -1 and padd last const to True
    if add_stop:
      self.stop=self.DR[:,1:] == -1
      self.stop = np.concatenate([self.stop,  np.ones((self.stop.shape[0], 1), dtype=bool)],axis=1)
    if add_mask:
      self.mask=self.DR != -1

    if standardize:
      dr_mean,dr_std=1.782,1.084
      kt_mean,kt_std=1.397,1.117
      self.DR=(self.DR-dr_mean)/dr_std
      self.kt=(self.kt-kt_mean)/kt_std

  def __getitem__(self, index):
    inputs=np.array([self.DR[index],self.kt[index]])
    if self.add_stop:
      inputs=np.concatenate([inputs,[self.stop[index]]],axis=0)
    self.data=torch.transpose(torch.tensor(inputs),0,1)

    if self.add_mask:
        return [self.data,self.mask[index]]
    else:
        return self.data

  def __len__(self):
    return len(self.DR)

class constit_dataset(torch.utils.data.Dataset):
  def __init__(self, file_path, NConstituents=20, add_stop=False):
    super(constit_dataset, self).__init__()
    self.data=torch.tensor([])
    self.add_stop=add_stop

    f = h5py.File(file_path,'r')
    es=f["constituents"]["E"]
    pxs=f["constituents"]["PX"]
    pys=f["constituents"]["PY"]
    pzs=f["constituents"]["PZ"]
    self.E=es[:,:NConstituents]
    self.px=pxs[:,:NConstituents]
    self.py=pys[:,:NConstituents]
    self.pz=pzs[:,:NConstituents]

    #Check if next is -1 and padd last const to True
    if add_stop:
      self.stop=self.E[:,1:] == 0
      self.stop = np.concatenate([self.stop,  np.ones((self.stop.shape[0], 1), dtype=bool)],axis=1)

  def __getitem__(self, index):
    inputs=np.array([self.E[index],self.px[index],self.py[index],self.pz[index]])
    if self.add_stop:
      inputs=np.concatenate([inputs,[self.stop[index]]],axis=0)
    self.data=torch.transpose(torch.tensor(inputs),0,1)

    return self.data

  def __len__(self):
    return len(self.E)

def get_loaders(input_format="ktdr",train_file=None,val_file=None,batch_size=256, num_workers=1, shuffle=True):

  if input_format=="ktdr":
    train_dataset = ktdr_dataset(train_file)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,)
    test_dataset = ktdr_dataset(val_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,)

  elif input_format=="4vec":
    train_dataset = constit_dataset(train_file)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,)
    test_dataset = constit_dataset(val_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,)
  return train_loader,test_loader

# ---------------------------------------------------------------------
# Schedulers and optimizers
# ---------------------------------------------------------------------
def get_lin_scheduler(num_epochs, num_batches, lr_decay, optimizer):
    training_steps = num_epochs * num_batches
    lr_fn = lambda step: 1.0 - ((1.0 - lr_decay) * (step / training_steps))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_fn)
    return scheduler

def get_exp_scheduler(
    num_epochs: int,
    num_batches: int,
    optimizer: torch.optim.Optimizer,
    final_reduction=1e-2,
):
    training_steps = num_epochs * num_batches
    lr_fn = lambda step: final_reduction ** (step / training_steps)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_fn)
    return scheduler

def get_cos_scheduler(num_epochs, num_batches, optimizer, eta_min=1e-6):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        # T_0=len(train_loader)//5,
        T_0=num_batches * num_epochs + 1,
        eta_min=eta_min,
    )
    return scheduler

def get_cos_damping_scheduler(
    optimizer,
    base_lr,
    num_epochs,
    cos_start_epoch=0,
    cos_end_epoch=None,
    cos_damping_final_lr=5e-5,
    cos_damping_amplitude=0.1,
    cos_damping_period_epochs=1.0,
):
    start_epoch = max(int(cos_start_epoch), 0)
    if cos_end_epoch is None:
        cos_end_epoch = num_epochs
    end_epoch = max(int(cos_end_epoch), start_epoch + 1)

    final_ratio = max(float(cos_damping_final_lr) / float(base_lr), 1e-12)
    period_epochs = max(float(cos_damping_period_epochs), 1e-12)

    def lr_lambda(epoch):
        if epoch < start_epoch:
            return 1.0
        if epoch >= end_epoch:
            return final_ratio

        progress = (epoch - start_epoch) / float(end_epoch - start_epoch)
        baseline = 1.0 + (final_ratio - 1.0) * progress
        phase = 2.0 * math.pi * (epoch - start_epoch) / period_epochs
        oscillation_factor = 1.0 + cos_damping_amplitude * math.cos(phase)
        return max(baseline * oscillation_factor, 1e-12)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def get_epoch_cosine_scheduler(optimizer, num_epochs, eta_min=1e-6):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(num_epochs), 1),
        eta_min=eta_min,
    )

def get_plateau_scheduler(
    optimizer,
    factor=0.5,
    patience=2,
    min_lr=1e-6,
):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )

def make_scheduler(args, optimizer):
    scheduler_name = getattr(args, "scheduler", "none")
    if scheduler_name == "cos_damping":
        return get_cos_damping_scheduler(
            optimizer=optimizer,
            base_lr=args.lr,
            num_epochs=args.epochs,
            cos_start_epoch=args.cos_damping_start_epoch,
            cos_end_epoch=args.cos_damping_end_epoch,
            cos_damping_final_lr=args.cos_damping_final_lr,
            cos_damping_amplitude=args.cos_damping_amplitude,
            cos_damping_period_epochs=args.cos_damping_period_epochs,
        )
    if scheduler_name == "cosine":
        return get_epoch_cosine_scheduler(
            optimizer=optimizer,
            num_epochs=args.epochs,
            eta_min=args.scheduler_min_lr,
        )
    if scheduler_name == "plateau":
        return get_plateau_scheduler(
            optimizer=optimizer,
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.scheduler_min_lr,
        )
    return None

def step_scheduler(scheduler, args, metric=None):
    if scheduler is None:
        return
    if getattr(args, "scheduler", "none") == "plateau":
        scheduler.step(metric)
    else:
        scheduler.step()

def make_optimizer(args, model):
    optimizer_name = getattr(args, "optimizer", "adam")
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

# ---------------------------------------------------------------------
# Functions to help with loading in/out config data
# ---------------------------------------------------------------------
def _config_get(args_or_dict, key, default=None):
    if isinstance(args_or_dict, dict):
        return args_or_dict.get(key, default)
    return getattr(args_or_dict, key, default)

def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)

def _as_int(value, default=None):
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("none", "null", ""):
            return default
        match = re.search(r"-?\d+", text)
        if match is None:
            return default
        return int(match.group(0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _as_float(value, default=None):
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("none", "null", ""):
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _epoch_display_from_index(epoch):
    epoch_idx = _as_int(epoch, None)
    if epoch_idx is None or epoch_idx < 0:
        return None
    return epoch_idx + 1

def resolved_model_mode(args_or_dict):
    if _as_bool(_config_get(args_or_dict, "cnf", False)):
        return "CNF"
    if _as_bool(_config_get(args_or_dict, "mdn", False)):
        return "MDN"
    return "Regression"

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------
def build_unbinned_model(input_dim, args_or_dict):
    get = lambda key, default=None: _config_get(args_or_dict, key, default)
    if _as_bool(get("cnf", False)):
        return models_generative.model_CNF(
            input_dim=input_dim[1]*input_dim[2],
            embed_dim=_as_int(get("embed_dim", 256), 256),
            num_heads=_as_int(get("num_heads", 1), 1),
            num_layers=_as_int(get("num_layers", 2), 2),
            ff_dim=_as_int(get("ff_dim", 128), 128),
            cnf_hidden=_as_int(get("cnf_hidden", get("flow_hidden", 128)), 128),
            #cnf_steps=_as_int(get("cnf_steps", 8), 8),
        )
    if _as_bool(get("fm", False)):
        return models_generative.model_FM(
            input_dim=input_dim[1]*input_dim[2],
            hidden_dim=_as_int(get("embed_dim", 256), 256),
            #num_heads=_as_int(get("num_heads", 1), 1),
            #num_layers=_as_int(get("num_layers", 2), 2),
            #ff_dim=_as_int(get("ff_dim", 128), 128),
            steps=_as_int(get("cnf_steps", 50), 50),
        )
    if _as_bool(get("mdn", False)):
        return models_generative.model_autoregressive_transformer_MDN(
            input_dim=input_dim[2],
            n_mix=_as_int(get("n_mix", 25), 25),
            embed_dim=_as_int(get("embed_dim", 256), 256),
            num_heads=_as_int(get("num_heads", 1), 1),
            num_layers=_as_int(get("num_layers", 2), 2),
            ff_dim=_as_int(get("ff_dim", 128), 128),
        )
    if _as_bool(get("nf", False)):
        return models_generative.model_normalizing_flow(
            input_dim=input_dim[1]*input_dim[2],
            num_flows=6,
            latent_dim=128,
        )
    if _as_bool(get("diff", False)):
        return models_generative.model_diffusion(input_dim=input_dim[1]*input_dim[2], hidden_dim=256)
    if _as_bool(get("sde", False)):
        return models_generative.model_score_SDE(input_dim=input_dim[1]*input_dim[2], hidden_dim=256)
    return models_generative.model_autoregressive_transformer(
        input_dim=input_dim[2],
        embed_dim=_as_int(get("embed_dim", 256), 256),
        num_heads=_as_int(get("num_heads", 1), 1),
        num_layers=_as_int(get("num_layers", 2), 2),
        ff_dim=_as_int(get("ff_dim", 128), 128),
    )

def save_model(model, log_dir, name):
    torch.save(model, os.path.join(log_dir, f"model_{name}.pt"))
    return

def load_model(model_path, map_location="cpu"):
    try:
        return torch.load(model_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=map_location)

# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------
def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    args,
    is_best=False,
    ckpt_name=None,
    train_losses=None,
    test_losses=None,
    loss_curves=None,
    lr_history=None,
    scheduler=None,
    best_epoch=None,
    best_loss=None,
    current_lr=None,
):
    save_mode = _config_get(args, "save_mode", "checkpoint")
    if save_mode == "none":
        return None
    if save_mode not in ("checkpoint", "model"):
        raise ValueError("--save-mode must be one of: checkpoint, model, none")

    ckpt_dir = os.path.join(_config_get(args, "log_dir"), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if ckpt_name is None:
        ckpt_name = f"epoch_{epoch:03d}.pt"

    args_dict = dict(args) if isinstance(args, dict) else vars(args).copy()
    best_epoch_display = _epoch_display_from_index(best_epoch)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    payload = {
        "artifact_type": "checkpoint" if save_mode == "checkpoint" else "model_state",
        "save_mode": save_mode,
        "epoch": epoch,
        "epoch_display": _epoch_display_from_index(epoch),
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "best_epoch": best_epoch,
        "best_epoch_display": best_epoch_display,
        "best_loss": best_loss,
        "args": args_dict,
        "model_mode": resolved_model_mode(args),
        "doCNF": _as_bool(_config_get(args, "cnf", False)),
        "doMDN": _as_bool(_config_get(args, "mdn", False)) and not _as_bool(_config_get(args, "cnf", False)),
        "doMixedLoss": _as_bool(_config_get(args, "mixed_loss", False)),
    }

    if save_mode == "checkpoint":
        payload.update(
            {
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scheduler": _config_get(args, "scheduler", "none"),
                "optimizer": _config_get(args, "optimizer", "adam"),
                "weight_decay": _config_get(args, "weight_decay", 0.0),
                "grad_clip": _config_get(args, "grad_clip", 0.0),
                "lr": _config_get(args, "lr", None),
                "current_lr": current_lr if current_lr is not None else (optimizer.param_groups[0]["lr"] if optimizer is not None else None),
                "next_lr": optimizer.param_groups[0]["lr"] if optimizer is not None else None,
                "train_losses": train_losses if train_losses is not None else [],
                "test_losses": test_losses if test_losses is not None else [],
                "loss_curves": loss_curves if loss_curves is not None else {},
                "lr_history": lr_history if lr_history is not None else [],
            }
        )

    torch.save(payload, ckpt_path)
    if is_best: torch.save(payload, "best.pt")
    saved_what = "checkpoint" if save_mode == "checkpoint" else "model state"
    print(f"Saved {saved_what} to {ckpt_path}", flush=True)
    return ckpt_path

def load_checkpoint(shape,args):
    if len(args.model_path) == 0:
      raise ValueError("--contin requires --model-path/--checkpoint")
    load_path = args.model_path[0] if isinstance(args.model_path, list) else args.model_path
    loaded = load_model(load_path)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
      resume_state = loaded
      loaded_args = loaded.get("args", {})
      for key in ( "cnf", "mdn", "mixed_loss", "embed_dim", "num_heads", "num_layers", "ff_dim", "n_mix", "cnf_hidden", "cnf_steps", "flow_hidden",):
        if key in loaded_args:
          setattr(args, key, loaded_args[key])
      model = build_unbinned_model(shape, args)
      model.load_state_dict(loaded["model_state_dict"])
    else:
      model = loaded
    return model,resume_state

# ---------------------------------------------------------------------
# Save/load loss and learning rates
# ---------------------------------------------------------------------
def save_loss_csv(epoch_losses=None, loss_curves=None, out_dir=""):
    if ((epoch_losses is None or len(epoch_losses) == 0) and
        (loss_curves is None or len(loss_curves) == 0)):
        return

    os.makedirs(out_dir, exist_ok=True)
    metric_names = []
    if epoch_losses is not None and len(epoch_losses) > 0:
        metric_names.append("self_loss")
    if loss_curves is not None:
        for metric_name, curve in loss_curves.items():
            if curve is not None and len(curve) > 0:
                metric_names.append(metric_name)

    max_len = len(epoch_losses) if epoch_losses is not None else 0
    if loss_curves is not None:
        for curve in loss_curves.values():
            if curve is not None:
                max_len = max(max_len, len(curve))

    csv_path = os.path.join(out_dir, "loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + metric_names)
        for i in range(max_len):
            row = [i + 1]
            for metric_name in metric_names:
                if metric_name == "self_loss":
                    value = epoch_losses[i] if i < len(epoch_losses) else ""
                else:
                    curve = loss_curves.get(metric_name, None) if loss_curves is not None else None
                    value = curve[i] if (curve is not None and i < len(curve)) else ""
                row.append(value)
            writer.writerow(row)
    print(f"Saved loss CSV to {csv_path}", flush=True)

def load_loss_history_csv(csv_path):
    if not os.path.exists(csv_path):
        return {}
    curves = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for name in reader.fieldnames or []:
            if name != "epoch":
                curves[name] = []
        for row in reader:
            for name in curves:
                value = row.get(name, "")
                curves[name].append(np.nan if value in (None, "") else float(value))
    return curves

def save_lr_csv(lr_history, out_dir=""):
    if lr_history is None or len(lr_history) == 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "lr_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr"])
        for i, lr in enumerate(lr_history):
            writer.writerow([i + 1, lr])
    print(f"Saved LR CSV to {csv_path}", flush=True)

def save_lr_plot(lr_history, out_dir=""):
    if lr_history is None or len(lr_history) == 0:
        return
    fig = plt.figure(figsize=(6.0, 4.0))
    plt.plot(range(1, len(lr_history) + 1), lr_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs Epoch")
    plt.grid(True)
    fig.savefig(os.path.join(out_dir, "lr_vs_epoch.png"))
    fig.savefig(os.path.join(out_dir, "lr_vs_epoch.pdf"))
    plt.close(fig)

def loss_plot(loss_train,loss_test,outdir="./Plots/", loss_curves=None):

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  train_arr = np.asarray(loss_train, dtype=float)
  test_arr = np.asarray(loss_test, dtype=float)
  n_epochs = min(len(train_arr), len(test_arr))
  train_arr = train_arr[:n_epochs]
  test_arr = test_arr[:n_epochs]
  finite = np.isfinite(train_arr) & np.isfinite(test_arr)
  truncated = False
  if np.any(~finite):
    first_bad = int(np.argmax(~finite))
    train_arr = train_arr[:first_bad]
    test_arr = test_arr[:first_bad]
    n_epochs = first_bad
    truncated = True

  epochs=range(1, n_epochs + 1)

  #Move to delta-loss if negative values
  min_train=np.min(loss_train)
  min_test=np.min(loss_test)
  if min_train<0 or min_test<0:
      min_total=np.min(min_train,min_test)
      loss_train+=min_total+1
      loss_test+=min_total+1

  fig,ax = plt.subplots(figsize=(6.0, 4.0))
  if n_epochs > 0:
    ax.plot(epochs, train_arr, marker="o", label="Train")
    ax.plot(epochs, test_arr, marker="o", label="Test")
  if truncated:
    ax.text( 0.5, 0.5, f"Loss history truncated at epoch {n_epochs}\nnext epoch contains nan/inf", ha="center", va="center", transform=ax.transAxes, fontsize=9,)
  ax.legend()
  ax.set_xlabel("Epoch")
  ax.set_ylabel("Loss")
  if min_train<0 or min_test<0:
      ax.set_ylabel("Delta*Loss+1")
  ax.set_yscale("log")
  ax.grid(True)
  fig.savefig(os.path.join(outdir,"loss_vs_epoch.png"))
  fig.savefig(os.path.join(outdir,"loss_vs_epoch.pdf"))
  fig.savefig(os.path.join(outdir,"loss_train_val_vs_epoch.png"))
  fig.savefig(os.path.join(outdir,"loss_train_val_vs_epoch.pdf"))
  plt.close(fig)

  save_loss_csv( epoch_losses=loss_train, loss_curves={"test_loss": loss_test, **(loss_curves or {})}, out_dir=outdir,)

# ---------------------------------------------------------------------
# Arguments and metadata saving
# ---------------------------------------------------------------------
def save_optimizer_states(optimizer, scheduler, scaler, log_dir): #not used
    torch.save(
        {
            "opt_state_dict": optimizer.state_dict(),
            "sched_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        os.path.join(log_dir, "opt_state_dict.pt"),
    )
    return

def load_optimizer_states(optimizer, scheduler, scaler, log_dir):
    state_dicts = torch.load(os.path.join(log_dir, "opt_state_dict.pt"))
    optimizer.load_state_dict(state_dicts["opt_state_dict"])
    scheduler.load_state_dict(state_dicts["sched_state_dict"])
    scaler.load_state_dict(state_dicts["scaler_state_dict"])
    return

def append_training_metadata(args, best_epoch=None, best_loss=None):
    txt_path = os.path.join(args.log_dir, "arguments.txt")
    with open(txt_path, "a") as f:
        if best_epoch is None:
            f.write("\n")
            f.write(f"{'resolved_model_mode':20s} {resolved_model_mode(args)}\n")
            f.write(f"{'save_mode':20s} {args.save_mode}\n")
            f.write(f"{'optimizer':20s} {args.optimizer}\n")
            f.write(f"{'weight_decay':20s} {args.weight_decay}\n")
            f.write(f"{'grad_clip':20s} {args.grad_clip}\n")
            f.write(f"{'scheduler':20s} {args.scheduler}\n")
            f.write(f"{'scheduler_min_lr':20s} {args.scheduler_min_lr}\n")
            f.write(f"{'plateau_factor':20s} {args.plateau_factor}\n")
            f.write(f"{'plateau_patience':20s} {args.plateau_patience}\n")
            f.write(f"{'cos_damping_start_epoch':20s} {args.cos_damping_start_epoch}\n")
            f.write(f"{'cos_damping_end_epoch':20s} {args.cos_damping_end_epoch if args.cos_damping_end_epoch is not None else args.epochs}\n")
            f.write(f"{'cos_damping_final_lr':20s} {args.cos_damping_final_lr}\n")
            f.write(f"{'cos_damping_amplitude':20s} {args.cos_damping_amplitude}\n")
            f.write(f"{'cos_damping_period_epochs':20s} {args.cos_damping_period_epochs}\n")
        else:
            best_epoch_display = _epoch_display_from_index(best_epoch)
            f.write("\n")
            f.write("# Best training result\n")
            f.write(f"best_epoch: {best_epoch_display}\n")
            f.write(f"best_epoch_display: {best_epoch_display}\n")
            f.write(f"best_epoch_index: {best_epoch}\n")
            f.write(f"best_loss: {best_loss}\n")
            if args.save_mode != "none":
                f.write("best_checkpoint: checkpoints/best.pt\n")

def save_arguments(args):
    tmp = args.log_dir
    if getattr(args, "contin", False):
        os.makedirs(tmp, exist_ok=True)
        with open(os.path.join(tmp, "arguments.txt"), "w") as f:
            arg_dict = vars(args)
            for k, v in arg_dict.items():
                f.write(f"{k:20s} {v}\n")
        return args

    i = 0
    while os.path.isdir(tmp):
        i += 1
        tmp = args.log_dir + f"_{i}"

    args.log_dir = tmp
    os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, "arguments.txt"), "w") as f:
        arg_dict = vars(args)
        for k, v in arg_dict.items():
            f.write(f"{k:20s} {v}\n")
    return args

def parse_input():
    """Parse_Input args. Defaults match the current hard-coded values."""
    p = argparse.ArgumentParser(description="Train transformer/MDN on Lund data")

    # data / io
    p.add_argument("--train-file", default="inputFiles/discretized/qcd_lund_cut_lundTree_kt_deltaR_train.h5", help="Path to training .h5")
    p.add_argument("--val-file", default="inputFiles/discretized/qcd_lund_cut_lundTree_kt_deltaR_val.h5", help="Path to validation .h5 (unused yet)")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p.add_argument("--num-workers", type=int, default=1, help="DataLoader workers")
    p.add_argument("--shuffle", action="store_true", default=True, help="Shuffle training loader (default: True)")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable shuffle")
    p.add_argument("--input_format", type=str, choices=["ktdr","4vec"], default="ktdr", help="What format of inputs we are using")

    # training
    p.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="Optimizer")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay")
    p.add_argument("--grad-clip", type=float, default=0.0, help="Gradient norm clipping. Set <=0 to disable")
    p.add_argument("--scheduler", type=str, default="none", choices=["none", "cos_damping", "cosine", "plateau"], help="Learning rate scheduler")
    p.add_argument("--scheduler-min-lr", type=float, default=1e-6, help="Minimum LR for cosine/plateau schedulers")
    p.add_argument("--plateau-factor", type=float, default=0.5, help="LR multiplier for --scheduler plateau")
    p.add_argument("--plateau-patience", type=int, default=2, help="Plateau epochs before reducing LR")
    p.add_argument("--cos-damping-start-epoch", type=int, default=0, help="Epoch where cosine damping starts")
    p.add_argument("--cos-damping-end-epoch", type=int, default=None, help="Epoch where cosine damping ends")
    p.add_argument("--cos-damping-final-lr", type=float, default=5e-5, help="Final LR for cosine damping")
    p.add_argument("--cos-damping-amplitude", type=float, default=0.0, help="Cosine oscillation amplitude")
    p.add_argument("--cos-damping-period-epochs", type=float, default=1.0, help="Cosine oscillation period in epochs")
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    p.add_argument("--seed", type=int, default=0, help="Random seed (overrides helpers_train default if you want)")

    # model switches
    p.add_argument("--mdn", action="store_true", default=False, help="Use MDN head (default: True)")
    p.add_argument("--no-mdn", dest="mdn", action="store_false", help="Disable MDN, use regression head")
    p.add_argument("--n-mix", type=int, default=25, help="Number of MDN mixtures")
    p.add_argument("--mixed-loss", action="store_true", default=False, help="Use mixed loss (default: False)")

    # CNF switch (overrides mdn/regression)
    p.add_argument("--cnf", action="store_true", default=False, help="Use Continuous Normalizing Flow head")
    p.add_argument("--cnf-hidden", type=int, default=128, help="CNF vector field hidden size")
    p.add_argument("--cnf-steps", type=int, default=8, help="CNF Euler steps for integration")

    # transformer hyperparams (keep defaults = your current test_model defaults)
    p.add_argument("--embed-dim", type=int, default=256, help="Transformer embedding dim")
    p.add_argument("--num-heads", type=int, default=1, help="Transformer num heads")
    p.add_argument("--num-layers", type=int, default=2, help="Transformer num layers")
    p.add_argument("--ff-dim", type=int, default=128, help="Transformer feedforward dim")

    #
    p.add_argument("--nf", action="store_true", default=False, help="Use NormFlows")
    p.add_argument("--fm", action="store_true", default=False, help="Use Continuous Normalizing Flow head")
    p.add_argument("--diff", action="store_true", default=False, help="Use Masked autoregressive flow")
    p.add_argument("--sde", action="store_true", default=False, help="Use Masked autoregressive flow")

    # auxiliary diagnostics
    p.add_argument("--flow-hidden", type=int, default=128, help="Hidden size for auxiliary flow nets")
    p.add_argument("--multi-loss-plot", action="store_true", default=False, help="Log multiple loss definitions without affecting main training")

    # misc
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="Force device; default auto")

    # logging / checkpointing
    p.add_argument("--log-dir", dest="log_dir", type=str, default="models/test",help="Logging directory")
    p.add_argument( "--plot-dir", "--plot-out-dir", dest="plot_dir", type=str, default=None, help="Output directory for plots. Default: use --log-dir / inferred checkpoint log directory",)
    p.add_argument("--save-mode", type=str, default="checkpoint", choices=["checkpoint", "model", "none"], help="Saved training artifact: full checkpoint, model-state only, or none")
    p.add_argument("--contin", action="store_true", default=False,help="Continue training from a saved model")
    p.add_argument("--model-path", "--checkpoint", dest="model_path", type=str, nargs="+",default=[],help="Path(s) to model/checkpoint to load")
    
    # plotting options
    p.add_argument( "--hist2d-xrange", type=float, nargs=2, default=None, help="2D Lund histogram x range: xmin xmax",)
    p.add_argument( "--hist2d-yrange", type=float, nargs=2, default=None, help="2D Lund histogram y range: ymin ymax",)
    p.add_argument( "--hist2d-bins", type=int, nargs=2, default=[20, 20], help="2D Lund histogram bins: xbins ybins",)
    p.add_argument( "--hist2d-shape", "--hist2d_shape", "--hist2d-layout", dest="hist2d_shape", type=int, nargs=2, default=None, metavar=("ROWS", "COLS"), help="Manual 2D Lund subplot shape. Default auto: 2 -> 1x2, 3 -> 1x3, 4 -> 2x2",)
    p.add_argument( "--plot-max-batches", type=int, default=None, help="Only plot this many validation batches. Default: plot all batches",)
    p.add_argument( "--hist1d-ranges", type=float, nargs="+", default=None, help="Flattened 1D ranges: kt_min kt_max dr_min dr_max",)
    p.add_argument( "--hist1d-bins", type=int, default=30, help="Number of bins for 1D histograms",)
    p.add_argument( "--hist1d-logy", action="store_true", default=True, help="Also save log-y 1D histograms",)
    p.add_argument( "--hist-ratio-diff", "--hist-diff-ratio", action="store_true", default=True, help="Also save generated-vs-original relative difference plots for multi-sample 1D/2D histograms",)
    p.add_argument( "--hist-ratio-min-count", type=int, default=5, help="Mask 2D relative-difference bins with fewer original entries than this",)
    p.add_argument( "--hist-ratio-vmax", type=float, default=1.0, help="Symmetric color limit for 2D fractional relative-difference plots",)

    return p.parse_args()
