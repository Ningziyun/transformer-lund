import os,sys
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt

import argparse

import torch
from torch.utils.data import DataLoader

import models_generative

import helpers_vqvae
import helpers

# ---------------------------------------------------------------------
# Data loaders for VQ-VAE codes (output of encode_jets.py)
# ---------------------------------------------------------------------
class codes_dataset(torch.utils.data.Dataset):
  """
  Token codes written by encode_jets.py: codes [n_jets, n_const, n_levels] (-1 = padded constituent) and,
  if add_conditional, cond [n_jets, 5] = [pt, eta, phi, m, N_const] (same vector as jet_data above).
  Returns codes (long) or [codes, cond].
  """
  def __init__(self, file_path, add_conditional=False):
    super(codes_dataset, self).__init__()
    self.add_conditional=add_conditional
 
    with h5py.File(file_path,'r') as f:
      self.codes=f["codes"][:].astype(np.int64)
      self.attrs={k: (v.item() if np.ndim(v)==0 and hasattr(v, "item") else v) for k, v in f.attrs.items()}
      if self.add_conditional:
        if "cond" not in f:
          raise ValueError(f"{file_path} has no 'cond' dataset (input file had no jet group): cannot run --conditional")
        self.cond=f["cond"][:].astype(np.float32)
 
    self.n_const=self.codes.shape[1]
    self.n_levels=self.codes.shape[2]
    self.mask=self.codes[..., 0] < 0 #True/False=Padded/unpadded
 
  def __getitem__(self, index):
    codes=torch.from_numpy(self.codes[index])
    if self.add_conditional:
      return [codes, torch.from_numpy(self.cond[index])]
    return codes
 
  def __len__(self):
    return len(self.codes)
 
def get_code_loaders(args):
  train_dataset = codes_dataset(args.train_codes, add_conditional=args.conditional)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)
  test_dataset = codes_dataset(args.val_codes, add_conditional=args.conditional)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
  return train_loader,test_loader
 
def format_codes(X, args, device):
  """codes (and conditioning) to device; same role as format_input for the original-format loaders"""
  if args.conditional:
    codes, jet = X
    return codes.to(device), jet.to(device)
  return X.to(device), None
 
def build_binned_model(args_or_dict, input_shape, cond_shape):

    if args_or_dict.architecture=="VQGPT":
        tokenizer, _ = helpers_vqvae.load_tokenizer_checkpoint(helpers._config_get(args_or_dict,"tokenizer"), "cpu")
        return models_generative.model_gpt_transformer(
            tokenizer,
            n_const=input_shape[1],
            cond_dim=cond_shape[-1],
            embed_dim=helpers._as_int(helpers._config_get(args_or_dict,"embed_dim")),
            num_heads=helpers._as_int(helpers._config_get(args_or_dict,"num_heads")),
            num_layers=helpers._as_int(helpers._config_get(args_or_dict,"num_layers")),
            ff_dim=helpers._as_int(helpers._config_get(args_or_dict,"ff_dim")),
            n_register=helpers._as_int(helpers._config_get(args_or_dict,"registers")),
            temperature=helpers._as_float(helpers._config_get(args_or_dict,"temperature"), 1.0),
            top_k=helpers._as_int(helpers._config_get(args_or_dict,"top_k")),
        )

def parse_input():
    """Parse_Input args. Defaults match the current hard-coded values."""
    parser = argparse.ArgumentParser(description="Train transformer/MDN on Lund data")

    # data / io
    parser.add_argument("--train-codes", default="inputFiles/discretized/qcd_lund_cut_lundTree_kt_deltaR_train.h5", help="Path to training .h5")
    parser.add_argument("--val-codes", default="inputFiles/discretized/qcd_lund_cut_lundTree_kt_deltaR_val.h5", help="Path to validation .h5 (unused yet)")
    parser.add_argument("--tokenizer", default="inputFiles/discretized/qcd_lund_cut_lundTree_kt_deltaR_val.h5", help="Path to validation .h5 (unused yet)")
    parser.add_argument("--val-file", default="inputFiles/discretized/qcd_lund_cut_lundTree_kt_deltaR_val.h5", help="Path to validation .h5 (unused yet)")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="Force device; default auto")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=1, help="DataLoader workers")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle training loader (default: True)")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable shuffle")
    parser.add_argument("--input_format", type=str, choices=["ktdr","4vec","relvec"], default="ktdr", help="What format of inputs we are using")
    parser.add_argument("--preprocess", type=str, default=None, choices=["log","linear","standardize","logstd","shiftpad"], help="Preprocess the output (default: None")
    parser.add_argument("--flatten", action="store_true", default=False, help="Flatten the energy during training (default: False)")
    parser.add_argument("--num-constituents", type=int, default=20, help="Number of constituents")

    # Architectures
    parser.add_argument("--architecture","-a",type=str, choices=["VQGPT"], help="Architecture to run")
    parser.add_argument("--conditional", action="store_true", default=False, help="Add conditional information (default: False)")
    parser.add_argument("--registers", type=int, default=0, help="Number of registers for transformers (default:0)")

    # transformer hyperparams (keep defaults = your current test_model defaults)
    parser.add_argument("--embed-dim", type=int, default=256, help="Transformer embedding dim")
    parser.add_argument("--num-heads", type=int, default=2, help="Transformer num heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer num layers")
    parser.add_argument("--ff-dim", type=int, default=256, help="Transformer feedforward dim")

     # logging / checkpointing
    parser.add_argument("--log-dir", dest="log_dir", type=str, default="models/test",help="Logging directory")
    parser.add_argument("--plot-dir", dest="plot_dir", type=str, default=None, help="Output directory for plots. Default: use --log-dir / inferred checkpoint log directory",)
    parser.add_argument("--save-mode", type=str, default="full", choices=["full", "model", "none"], help="Saved training artifact: full info, model-state only, or none")
    parser.add_argument("--model-checkpoint", "--checkpoint", type=str, nargs="+",default=[],help="Path(s) to model/checkpoint to load")

    # training
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Optimizer weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping. Set <=0 to disable")
    parser.add_argument("--scheduler", type=str, default="warmup_cosine", choices=["none", "cos_damping", "cosine", "warmup_cosine", "plateau"], help="Learning rate scheduler")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6, help="Minimum LR for cosine/plateau schedulers")
    parser.add_argument("--plateau-factor", type=float, default=0.5, help="LR multiplier for --scheduler plateau")
    parser.add_argument("--plateau-patience", type=int, default=2, help="Plateau epochs before reducing LR")
    parser.add_argument("--cos-damping-start-epoch", type=int, default=0, help="Epoch where cosine damping starts")
    parser.add_argument("--cos-damping-end-epoch", type=int, default=None, help="Epoch where cosine damping ends")
    parser.add_argument("--cos-damping-final-lr", type=float, default=5e-5, help="Final LR for cosine damping")
    parser.add_argument("--cos-damping-amplitude", type=float, default=0.0, help="Cosine oscillation amplitude")
    parser.add_argument("--cos-damping-period-epochs", type=float, default=1.0, help="Cosine oscillation period in epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (overrides helpers_train default if you want)")
    parser.add_argument("--test", action="store_true", default=False, help="Setup with reduced training size for easy debugging/testing")

    # misc
    parser.add_argument("--multi-loss-plot", action="store_true", default=False, help="Log multiple loss definitions without affecting main training")
    parser.add_argument("--mask-bad", action="store_true", default=False, help="Mask bad generated consitiuents")

    # plotting options
    parser.add_argument("--validation-size",type=int, default=10000, help="How many images to generate and make plots for")
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

    #NOTE preprocessing is applied in the lunddata script now
    if args.input_format=="4vec" or args.input_format=="relvec":
        args.preprocess="logstd"

    return args
