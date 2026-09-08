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
from torch.utils.data import DataLoader
import torch.nn.functional as F

import models_generative

import helpers

# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
class ktdr_dataset(torch.utils.data.Dataset):
  def __init__(self, file_path, NConstituents=20, add_mask=False, preprocess=None):
    super(ktdr_dataset, self).__init__()
    self.data=torch.tensor([])
    self.add_mask=add_mask
    self.preprocess=preprocess

    f = h5py.File(file_path,'r')
    kts=f["lundplane"]["kt"]
    drs=f["lundplane"]["dr"]
    self.kt=kts[:,:NConstituents]
    self.DR=drs[:,:NConstituents]

    #Check if next is -1 and pad last const to True
    if add_mask:
      self.mask=self.DR == -1

  def __getitem__(self, index):
    inputs=np.array([self.kt[index],self.DR[index]])
    self.data=torch.transpose(torch.tensor(inputs),0,1)

    if self.preprocess:
        helpers.preprocess(self.data,"ktdr",self.preprocess)

    if self.add_mask:
        return [self.data,self.mask[index]]
    else:
        return self.data

  def __len__(self):
    return len(self.DR)

class constit_dataset(torch.utils.data.Dataset):
  def __init__(self, file_path, NConstituents=20, add_mask=False, preprocess=None):
    super(constit_dataset, self).__init__()
    self.data=torch.tensor([])
    self.add_mask=add_mask
    self.preprocess=preprocess

    f = h5py.File(file_path,'r')
    es=f["constituents"]["E"]
    pxs=f["constituents"]["PX"]
    pys=f["constituents"]["PY"]
    pzs=f["constituents"]["PZ"]
    self.e=es[:,:NConstituents]
    self.px=pxs[:,:NConstituents]
    self.py=pys[:,:NConstituents]
    self.pz=pzs[:,:NConstituents]

    #Check if next is -1 and pad last const to True
    if add_mask:
      self.mask=self.e == -1

  def __getitem__(self, index):
    inputs=np.array([self.e[index],self.px[index],self.py[index],self.pz[index]])
    self.data=torch.transpose(torch.tensor(inputs),0,1)

    if self.preprocess:
        helpers.preprocess(self.data,"4vec",self.preprocess)

    if self.add_mask:
        return [self.data,self.mask[index]]
    else:
        return self.data

  def __len__(self):
    return len(self.e)

class constit_relative_dataset(torch.utils.data.Dataset):
  def __init__(self, file_path, NConstituents=20, add_mask=False, preprocess=None):
    super(constit_relative_dataset, self).__init__()
    self.data=torch.tensor([])
    self.add_mask=add_mask
    self.preprocess=preprocess

    f = h5py.File(file_path,'r')
    ptfracs=f["relative_constituents"]["ptfrac"]
    detas=f["relative_constituents"]["deta"]
    dphis=f["relative_constituents"]["dphi"]
    ms=f["relative_constituents"]["m"]
    self.ptfrac=ptfracs[:,:NConstituents]
    self.deta=detas[:,:NConstituents]
    self.dphi=dphis[:,:NConstituents]
    self.m=ms[:,:NConstituents]

    self.jet_E=f["jet"]["E"]
    self.jet_px=f["jet"]["PX"]
    self.jet_py=f["jet"]["PY"]
    self.jet_pz=f["jet"]["PZ"]

    #Check if next is -1 and pad last const to True
    if add_mask:
      self.mask=self.ptfrac == -1

  def __getitem__(self, index):

    inputs=np.array([self.ptfrac[index],self.deta[index],self.dphi[index]])
    self.data=torch.transpose(torch.tensor(inputs),0,1)

    eps=1e-12
    jet_p = np.sqrt(self.jet_px[index]**2 + self.jet_py[index]**2 + self.jet_pz[index]**2)
    jet_pt = np.hypot(self.jet_px[index], self.jet_py[index])
    jet_eta = np.arcsinh(self.jet_pz[index] / np.maximum(jet_pt, eps))
    jet_phi = np.arctan2(self.jet_py[index], self.jet_px[index])
    jet_m = np.sqrt(np.maximum(self.jet_E[index]**2 - jet_p**2, 0.0))
    jet_data=torch.tensor([jet_pt,jet_eta,jet_phi,jet_m])

    if self.add_mask:
        return [self.data,jet_data,self.mask[index]]
    else:
        return [self.data,jet_data]

  def __len__(self):
    return len(self.ptfrac)

def get_loaders(args):

  if args.input_format=="ktdr":
    train_dataset = ktdr_dataset(args.train_file, NConstituents=args.num_constituents, add_mask=args.mixed_loss, preprocess=args.preprocess)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)
    test_dataset = ktdr_dataset(args.val_file, NConstituents=args.num_constituents, add_mask=args.mixed_loss, preprocess=args.preprocess)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)

  elif args.input_format=="4vec":
    train_dataset = constit_dataset(args.train_file, NConstituents=args.num_constituents, add_mask=args.mixed_loss, preprocess=args.preprocess)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)
    test_dataset = constit_dataset(args.val_file, NConstituents=args.num_constituents, add_mask=args.mixed_loss, preprocess=args.preprocess)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)

  elif args.input_format=="relvec":
    train_dataset = constit_relative_dataset(args.train_file, NConstituents=args.num_constituents, add_mask=args.mixed_loss, preprocess=args.preprocess)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)
    test_dataset = constit_relative_dataset(args.val_file, NConstituents=args.num_constituents, add_mask=args.mixed_loss, preprocess=args.preprocess)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)

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

def get_cosine_scheduler(optimizer, num_epochs, eta_min=1e-6):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(num_epochs), 1),
        eta_min=eta_min,
    )

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_epochs, min_lr, max_lr,):
    """
    Linear warm-up followed by cosine decay.
    """

    if num_epochs <= num_warmup_steps:
        raise ValueError("num_epochs must be greater than num_warmup_steps")

    if min_lr < 0:
        raise ValueError("min_lr must be >= 0")

    if max_lr <= 0:
        raise ValueError("max_lr must be > 0")

    if min_lr > max_lr:
        raise ValueError("min_lr must be <= max_lr")

    # LambdaLR multiplies the optimizer's initial LR, so set the optimizer LR to max_lr.
    if any(param_group["lr"] != max_lr for param_group in optimizer.param_groups):
        raise ValueError("optimizer learning rate must be set to max_lr")

    def lr_lambda(current_step):

        # Linear warm-up from min_lr -> max_lr
        if current_step < num_warmup_steps:
            lin_ratio = current_step / num_warmup_steps # 0 -> 1
            return (min_lr + (max_lr - min_lr) * lin_ratio) / max_lr # Convert absolute LR range to a multiplier.

        # Cosine decay from max_lr -> min_lr
        else:
            progress = ( current_step - num_warmup_steps) / max( 1, num_epochs - num_warmup_steps,)
            progress = min(progress, 1.0)
            cosine = 0.5 * ( 1.0 + math.cos(math.pi * progress)) # 0 -> 1
            return (min_lr + (max_lr - min_lr) * cosine) / max_lr # Convert absolute LR range to a multiplier.

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        return get_cosine_scheduler(
            optimizer=optimizer,
            num_epochs=args.epochs,
            eta_min=args.scheduler_min_lr,
        )
    if scheduler_name == "warmup_cosine":
        return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(int(0.2*args.epochs)-1, 0),
                num_epochs=args.epochs,
                min_lr=args.scheduler_min_lr,
                max_lr=args.lr,
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

def _config_get(args_or_dict, key, default=None):
    if isinstance(args_or_dict, dict):
        return args_or_dict.get(key, default)
    return getattr(args_or_dict, key, default)

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------
def build_unbinned_model(input_dim, args_or_dict):
    if args_or_dict.architecture=="MDN":
        pad_value=-1
        max_value=10
        if args_or_dict.input_format=="4vec":
            if args_or_dict.preprocess==None:
                max_value=3e3
            elif args_or_dict.preprocess=="standardize":
                max_value=50
                pad_value=-10
            elif args_or_dict.preprocess=="linearize":
                max_value=1
            elif args_or_dict.preprocess=="log":
                max_value=11
                pad_value=-10

        return models_generative.model_autoregressive_transformer_MDN(
            input_dim=input_dim[2],
            n_mix=_as_int(_config_get(args_or_dict,"MDN_nmix")),
            embed_dim=_as_int(_config_get(args_or_dict,"embed_dim")),
            num_heads=_as_int(_config_get(args_or_dict,"num_heads")),
            num_layers=_as_int(_config_get(args_or_dict,"num_layers")),
            ff_dim=_as_int(_config_get(args_or_dict,"ff_dim")),
            multi_head=_as_bool(_config_get(args_or_dict,"mixed_loss")),
            max_range=max_value,
            pad_value=pad_value,
        )
    elif args_or_dict.architecture=="CNF":
        return models_generative.model_CNF(
            input_dim=input_dim[1]*input_dim[2],
            embed_dim=_as_int(_config_get(args_or_dict,"embed_dim")),
            num_heads=_as_int(_config_get(args_or_dict,"num_heads")),
            num_layers=_as_int(_config_get(args_or_dict,"num_layers")),
            ff_dim=_as_int(_config_get(args_or_dict,"ff_dim")),
            cnf_hidden=_as_int(_config_get(args_or_dict,"cnf_hidden")),
            steps=_as_int(_config_get(args_or_dict,"cnf_steps")),
            time_dim=_as_int(_config_get(args_or_dict,"time_dim")),
        )
    elif args_or_dict.architecture=="FM":
        return models_generative.model_FM(
            input_dim=input_dim[1]*input_dim[2],
            hidden_dim=_as_int(_config_get(args_or_dict,"embed_dim")),
            steps=_as_int(_config_get(args_or_dict,"fm_steps")),
            time_dim=_as_int(_config_get(args_or_dict,"time_dim")),
        )
    elif args_or_dict.architecture=="NF":
        return models_generative.model_normalizing_flow(
            input_dim=input_dim[1]*input_dim[2],
            num_flows=_as_int(_config_get(args_or_dict,"nflows")),
            latent_dim=_as_int(_config_get(args_or_dict,"embed_dim")),
            flow_type=args_or_dict.nf_type,
        )
    elif args_or_dict.architecture=="Diffusion":
        return models_generative.model_diffusion(
                input_dim=input_dim[1]*input_dim[2], 
                hidden_dim=_as_int(_config_get(args_or_dict,"embed_dim")),
                time_steps=_as_int(_config_get(args_or_dict,"diff_steps")),
                beta_start=_as_float(_config_get(args_or_dict,"diff_beta_start")),
                beta_end=_as_float(_config_get(args_or_dict,"diff_beta_start")),
                mode=args_or_dict.diff_type,
                time_dim=_as_int(_config_get(args_or_dict,"time_dim")),
        )
    elif args_or_dict.architecture=="SDE":
        return models_generative.model_score_SDE(
                input_dim=input_dim[1]*input_dim[2],
                hidden_dim=_as_int(_config_get(args_or_dict,"embed_dim")),
                beta_min=_as_float(_config_get(args_or_dict,"sde_beta_min")),
                beta_max=_as_float(_config_get(args_or_dict,"sde_beta_max")),
                mode=args_or_dict.sde_type,
        )
    elif args_or_dict.architecture=="Transformer":
        return models_generative.model_autoregressive_transformer(
            input_dim=input_dim[2],
            embed_dim=_as_int(_config_get(args_or_dict,"embed_dim")),
            num_heads=_as_int(_config_get(args_or_dict,"num_heads")),
            num_layers=_as_int(_config_get(args_or_dict,"num_layers")),
            ff_dim=_as_int(_config_get(args_or_dict,"ff_dim")),
            multi_head=_as_bool(_config_get(args_or_dict,"mixed_loss")),
        )

def save_model(model, log_dir, name):
    torch.save(model, os.path.join(log_dir, f"model_{name}.pt"))
    return

def load_model(model_path, map_location="cpu"):
    #Note that torch load just unpickles a dictionary, which can be our checkpoint. The "model_state_dict" key holds the weights
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

    #From args figure out how much to save
    save_mode = _config_get(args, "save_mode", "full")
    if save_mode == "none":
        return None
    if save_mode not in ("full", "model"):
        raise ValueError("--save-mode must be one of: optimizer, model, none")

    #Get the path
    ckpt_dir = os.path.join(_config_get(args, "log_dir"), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if ckpt_name is None:
        ckpt_name = f"epoch_{epoch:03d}.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    #Save the model info
    args_dict = dict(args) if isinstance(args, dict) else vars(args).copy()
    checkpoint_info = {
        "save_mode": save_mode,
        "epoch": epoch,
        "loss": loss,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "args": args_dict,
        "model_state_dict": model.state_dict(),
    }

    #Add in the extra optimizer and learning info
    if save_mode == "full":
        checkpoint_info.update(
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

    #Actually save it
    torch.save(checkpoint_info, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}", flush=True)
    if is_best:  # if best save to it's own file
        best_path=os.path.dirname(ckpt_path) +"/best.pt"
        torch.save(checkpoint_info, best_path)
        print(f"Saved new best checkpoint to {best_path}", flush=True)
    return ckpt_path

def load_checkpoint(args):
    if len(args.model_path) == 0:
      raise ValueError("load_checkpoint requires --model-path")
    load_path = args.model_path[0] if isinstance(args.model_path, list) else args.model_path
    checkpoint_info = load_model(load_path)

    return checkpoint_info

def load_checkpoint_args(args, ignore_args=[], checkpoint_model=None):
    if checkpoint_model==None:
        checkpoint_info = load_checkpoint(args)

    #Load all the arguments
    checkpoint_info_args = checkpoint_info.get("args", {})
    for key in vars(args):
      if key in checkpoint_info_args:
        if key in ignore_args: continue
        setattr(args, key, checkpoint_info_args[key])

    return checkpoint_info

def load_checkpoint_model(shape, args, checkpoint_model=None):
    if checkpoint_model==None:
        checkpoint_info = load_checkpoint(args)

    model = build_unbinned_model(shape, args)
    model.load_state_dict(checkpoint_info["model_state_dict"])

    return model

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

def load_loss_csv(csv_path):
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
    print(f"Plotting learning rate to {out_dir}/lr_vs_epoch.pdf")

def loss_plot(loss_train,loss_test,out_dir="./Plots/", loss_curves=None):

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

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
  min_train=np.min(train_arr)
  min_test=np.min(test_arr)
  if min_train<0 or min_test<0:
      min_total=min(min_train,min_test)-1
      train_arr-=min_total
      test_arr-=min_total

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
  fig.savefig(os.path.join(out_dir,"loss_vs_epoch.png"))
  fig.savefig(os.path.join(out_dir,"loss_vs_epoch.pdf"))
  plt.close(fig)

  print(f"Plotting loss to {out_dir}/loss_vs_epoch.pdf")
  save_loss_csv( epoch_losses=loss_train, loss_curves={"test_loss": loss_test, **(loss_curves or {})}, out_dir=out_dir,)

# ---------------------------------------------------------------------
# Arguments and metadata saving
# ---------------------------------------------------------------------
def append_result_metadata(args, results):
    txt_path = os.path.join(args.log_dir, "arguments.txt")
    with open(txt_path, "a") as f:
        f.write("\n")
        for k, v in results.items():
            f.write(f"{k:20s} {v}\n")

def save_argument_metadata(args):
    #If continuing
    if getattr(args, "contin", False):
        os.makedirs(args.log_dir, exist_ok=True)
        with open(os.path.join(args.log_dir, "arguments.txt"), "w") as f:
            arg_dict = vars(args)
            for k, v in arg_dict.items():
                f.write(f"{k:20s} {v}\n")
        return

    #enumerate up the output directory if it exists
    log_dir = args.log_dir
    i = 0
    while os.path.isdir(log_dir):
        i += 1
        log_dir = args.log_dir + f"_{i}"
    if args.plot_dir==args.log_dir: args.plot_dir=log_dir
    args.log_dir = log_dir

    #Loop over arguments and save them
    os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, "arguments.txt"), "w") as f:
        arg_dict = vars(args)
        for k, v in arg_dict.items():
            f.write(f"{k:20s} {v}\n")

    return

def parse_input():
    """Parse_Input args. Defaults match the current hard-coded values."""
    parser = argparse.ArgumentParser(description="Train transformer/MDN on Lund data")

    # data / io
    parser.add_argument("--train-file", default="inputFiles/discretized/qcd_lund_cut_lundTree_kt_deltaR_train.h5", help="Path to training .h5")
    parser.add_argument("--val-file", default="inputFiles/discretized/qcd_lund_cut_lundTree_kt_deltaR_val.h5", help="Path to validation .h5 (unused yet)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=1, help="DataLoader workers")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle training loader (default: True)")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable shuffle")
    parser.add_argument("--input_format", type=str, choices=["ktdr","4vec","relvec"], default="ktdr", help="What format of inputs we are using")
    parser.add_argument("--preprocess", type=str, default=None, help="Preprocess the output (default: lNone")
    parser.add_argument("--flatten", action="store_true", default=False, help="Flatten the energy during training (default: False)")
    parser.add_argument("--num-constituents", type=int, default=20, help="Number of constituents")

    # Architectures
    parser.add_argument("--architecture","-a",type=str, choices=["Transformer","MDN","NF","CNF","Diffusion","SDE","FM"], help="Architecture to run")
    parser.add_argument("--mixed-loss", action="store_true", default=False, help="Use mixed loss (default: False)")

    # training
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Optimizer weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping. Set <=0 to disable")
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cos_damping", "cosine", "warmup_cosine", "plateau"], help="Learning rate scheduler")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6, help="Minimum LR for cosine/plateau schedulers")
    parser.add_argument("--plateau-factor", type=float, default=0.5, help="LR multiplier for --scheduler plateau")
    parser.add_argument("--plateau-patience", type=int, default=2, help="Plateau epochs before reducing LR")
    parser.add_argument("--cos-damping-start-epoch", type=int, default=0, help="Epoch where cosine damping starts")
    parser.add_argument("--cos-damping-end-epoch", type=int, default=None, help="Epoch where cosine damping ends")
    parser.add_argument("--cos-damping-final-lr", type=float, default=5e-5, help="Final LR for cosine damping")
    parser.add_argument("--cos-damping-amplitude", type=float, default=0.0, help="Cosine oscillation amplitude")
    parser.add_argument("--cos-damping-period-epochs", type=float, default=1.0, help="Cosine oscillation period in epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (overrides helpers_train default if you want)")
    parser.add_argument("--test", action="store_true", default=False, help="Setup with reduced training size for easy debugging/testing")

    # general model parameters
    parser.add_argument("--embed-dim", type=int, default=256, help="Transformer embedding dim")
    parser.add_argument("--time-dim", type=int, default=64, help="Transformer embedding dim")

    # transformer hyperparams (keep defaults = your current test_model defaults)
    parser.add_argument("--num-heads", type=int, default=2, help="Transformer num heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer num layers")
    parser.add_argument("--ff-dim", type=int, default=256, help="Transformer feedforward dim")

    # MDN parameters
    parser.add_argument("--MDN-nmix", type=int, default=25, help="Number of MDN mixtures")

    # CNF paramaters
    parser.add_argument("--cnf-hidden", type=int, default=128, help="CNF vector field hidden size")
    parser.add_argument("--cnf-steps", type=int, default=25, help="CNF Euler steps for integration")

    # FM parameters
    parser.add_argument("--fm-steps", type=int, default=25, help="FM Euler steps for integration")

    # NF paramaters
    parser.add_argument("--nflows", type=int, default=10, help="NF steps")
    parser.add_argument("--flow-type", type=str, default="neuralspline_coupling", choices=["realnvp","maf","neuralspline_coupling","neuralspline_autoregressive"], help="Algorithm for the normalizing flows")

    # Diff paramaters
    parser.add_argument("--diff-steps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--diff-beta-start", type=float, default=1e-4, help="Diffusion beta start")
    parser.add_argument("--diff-beta-end", type=float, default=2e-2, help="Diffusion beta end")
    parser.add_argument("--diff-type", type=str, default="DDPM", choices=["DDPM","DDIM"], help="Algorithm for the diffusion")

    # SDE parameters
    parser.add_argument("--SDE-beta-min", type=float, default=0.1, help="SDE beta min")
    parser.add_argument("--SDE-beta-max", type=float, default=20.0, help="SDE beta max")
    parser.add_argument("--SDE-type", type=str, default="sde", choices=["sde","ode"], help="Algorithm for the SDE")

    # misc
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="Force device; default auto")
    parser.add_argument("--multi-loss-plot", action="store_true", default=False, help="Log multiple loss definitions without affecting main training")
    parser.add_argument("--mask-bad", action="store_true", default=False, help="Mask bad generated consitiuents")

    # logging / checkpointing
    parser.add_argument("--log-dir", dest="log_dir", type=str, default="models/test",help="Logging directory")
    parser.add_argument("--plot-dir", dest="plot_dir", type=str, default=None, help="Output directory for plots. Default: use --log-dir / inferred checkpoint log directory",)
    parser.add_argument("--save-mode", type=str, default="full", choices=["full", "model", "none"], help="Saved training artifact: full info, model-state only, or none")
    parser.add_argument("--contin", action="store_true", default=False,help="Continue training from a saved model")
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", type=str, nargs="+",default=[],help="Path(s) to model/checkpoint to load")
    
    # plotting options
    parser.add_argument("--validation-size",type=int, default=100000, help="How many images to generate and make plots for")
    parser.add_argument("--hist2d-xrange", type=float, nargs=2, default=[-1,10], help="2D Lund histogram x range: xmin xmax",)
    parser.add_argument("--hist2d-yrange", type=float, nargs=2, default=[-5,7], help="2D Lund histogram y range: ymin ymax",)
    parser.add_argument("--hist2d-bins", type=int, nargs=2, default=[20, 20], help="2D Lund histogram bins: xbins ybins",)
    parser.add_argument("--hist2d-shape", "--hist2d_shape", "--hist2d-layout", dest="hist2d_shape", type=int, nargs=2, default=None, metavar=("ROWS", "COLS"), help="Manual 2D Lund subplot shape. Default auto: 2 -> 1x2, 3 -> 1x3, 4 -> 2x2",)
    parser.add_argument("--hist1d-ranges", type=float, nargs="+", default=None, help="Flattened 1D ranges: kt_min kt_max dr_min dr_max",)
    parser.add_argument("--hist1d-bins", type=int, default=30, help="Number of bins for 1D histograms",)
    parser.add_argument("--hist-ratio-min-count", type=int, default=5, help="Mask 2D relative-difference bins with fewer original entries than this",)
    parser.add_argument("--hist-ratio-vmax", type=float, default=1.0, help="Symmetric color limit for 2D fractional relative-difference plots",)

    #Some manipulations
    args=parser.parse_args()
    if args.plot_dir is None:
        args.plot_dir = args.log_dir

    return args
