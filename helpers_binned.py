import numpy as np
import h5py
import matplotlib.pyplot as plt

import argparse

import torch
from torch.utils.data import DataLoader

import models_generative
import models_vqvae

import helpers

# ---------------------------------------------------------------------
# Data loaders for VQ-VAE codes 
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
      self.codes=f["constituents"]["codes"][:].astype(np.int64)
      self.attrs={k: (v.item() if np.ndim(v)==0 and hasattr(v, "item") else v) for k, v in f.attrs.items()}
      if self.add_conditional:
        if "jet" not in f:
          raise ValueError(f"{file_path} has no 'cond' dataset (input file had no jet group): cannot run --conditional")
        j = f["jet"]
        self.jet = np.stack([np.asarray(j[k][:], dtype=np.float64) for k in ["Pt","Eta","Phi","M"]], axis=-1)
 
    self.n_const=self.codes.shape[1]
    self.n_levels=self.codes.shape[2]
    self.mask=self.codes[..., 0] < 0 #True/False=Padded/unpadded
 
  def __getitem__(self, index):
    codes=torch.from_numpy(self.codes[index])
    if self.add_conditional:
      return [codes, torch.from_numpy(self.jet[index])]
    return codes
 
  def __len__(self):
    return len(self.codes)
 
def get_code_loaders(args):
    train_dataset = codes_dataset(args.train_codes, add_conditional=(args.conditional or args.input_format=="relvec"))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)
    test_dataset = codes_dataset(args.test_codes, add_conditional=(args.conditional or args.input_format=="relvec"))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    return train_loader,test_loader
 
def format_codes(X, args, device):
    """codes (and conditioning) to device; same role as format_input for the original-format loaders"""
    if args.conditional or args.input_format=="relvec":
        codes, jet = X
        return codes.to(device), jet.to(device)
    return X.to(device), None

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------
def build_binned_model(args_or_dict, input_shape, cond_shape):
    if args_or_dict.architecture=="VQGPT":
        tokenizer, _ = models_vqvae.load_tokenizer_checkpoint(helpers.config_get(args_or_dict,"tokenizer"), "cpu")
        return models_generative.model_gpt_transformer(
            tokenizer,
            n_const=input_shape[1],
            cond_dim=cond_shape[-1],
            embed_dim=helpers.as_int(helpers.config_get(args_or_dict,"embed_dim")),
            num_heads=helpers.as_int(helpers.config_get(args_or_dict,"num_heads")),
            num_layers=helpers.as_int(helpers.config_get(args_or_dict,"num_layers")),
            ff_dim=helpers.as_int(helpers.config_get(args_or_dict,"ff_dim")),
            n_register=helpers.as_int(helpers.config_get(args_or_dict,"registers")),
            temperature=helpers.as_float(helpers.config_get(args_or_dict,"temperature"), 1.0),
            top_k=helpers.as_int(helpers.config_get(args_or_dict,"top_k")),
        )

# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
def parse_input():
    """Parse_Input args. Defaults match the current hard-coded values."""
    parser = argparse.ArgumentParser(description="Train transformer/MDN on Lund data")

    helpers.common_training_args(parser)

    # data
    parser.add_argument("--train-codes", help="Path to training codebook .h5")
    parser.add_argument("--test-codes", help="Path to testing codebook .h5 ")
    parser.add_argument("--tokenizer", help="Path to tokenizer .pkl")
    parser.add_argument("--val-file", help="Path to validation .h5 with original non-tokenized constituents")

    # Architectures
    parser.add_argument("--architecture","-a",type=str, choices=["VQGPT"], help="Architecture to run")
    parser.add_argument("--conditional", action="store_true", default=False, help="Add conditional information (default: False)")
    parser.add_argument("--registers", type=int, default=0, help="Number of registers for transformers (default:0)")

    # transformer hyperparams (keep defaults = your current test_model defaults)
    parser.add_argument("--embed-dim", type=int, default=256, help="Transformer embedding dim")
    parser.add_argument("--num-heads", type=int, default=2, help="Transformer num heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer num layers")
    parser.add_argument("--ff-dim", type=int, default=256, help="Transformer feedforward dim")

    #Some manipulations
    args=parser.parse_args()
    if args.plot_dir is None:
        args.plot_dir = args.log_dir

    #NOTE preprocessing is applied in the lunddata script now
    if args.input_format=="4vec" or args.input_format=="relvec":
        args.preprocess="logstd"

    return args
