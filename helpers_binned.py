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
class codes_stream_dataset(helpers.block_stream_dataset):
  """
  Token codes written by encode_vqvae.py, streamed from disk like the original-format loaders (helpers.block_stream_dataset):
    codes [n_jets, n_const, n_levels] (-1 = padded constituent), yielded as int64
    cond  [n_jets, 5] = [pt, eta, phi, m, N_const] float32 (same vector as jet_data of the original-format loader),
          only if add_conditional
  Each batch is codes, or [codes, cond].

  Codes files written before the "cond" dataset existed are read too: the conditioning is then built from
  jet/{Pt,Eta,Phi,M} plus N_const counted from the codes.
  """
  def __init__(self, file_path, add_conditional=False, batch_size=256, shuffle=True, block_size=None,
               blocks_per_buffer=32, drop_last=False, max_jets=None, num_workers=0):
    self.add_conditional=add_conditional

    with h5py.File(file_path,'r') as f:
      shape=f["constituents"]["codes"].shape
      self.attrs={k: (v.item() if np.ndim(v)==0 and hasattr(v, "item") else v) for k, v in f.attrs.items()}
      self.has_cond="cond" in f
      has_jet="jet" in f and all(k in f["jet"] for k in ["Pt","Eta","Phi","M"])
    if add_conditional and not (self.has_cond or has_jet):
      raise ValueError(f"{file_path} has no 'cond' dataset or jet group: cannot run --conditional")

    self.n_const=shape[1]
    self.n_levels=shape[2]
    self._cond_cache=None
    super(codes_stream_dataset, self).__init__(file_path, shape[0], shape[1]*shape[2]*4, batch_size=batch_size,
        shuffle=shuffle, block_size=block_size, blocks_per_buffer=blocks_per_buffer, drop_last=drop_last,
        max_jets=max_jets, num_workers=num_workers)

  def _read(self, f, blocks, need_cond=None):
    """(start, stop) row ranges -> codes [n, C, Q] int32, cond [n, 5] float32 or None"""
    need_cond = self.add_conditional if need_cond is None else need_cond
    n = sum(e-s for s, e in blocks)
    codes = np.empty((n, self.n_const, self.n_levels), dtype=np.int32)
    cond = np.empty((n, 5), dtype=np.float32) if need_cond else None
    pos = 0
    for s, e in blocks:
      codes[pos:pos+e-s] = f["constituents"]["codes"][s:e]
      if cond is not None:
        if self.has_cond:
          cond[pos:pos+e-s] = f["cond"][s:e]
        else: #older codes files: jet kinematics + multiplicity from the codes
          for j, k in enumerate(["Pt","Eta","Phi","M"]):
            cond[pos:pos+e-s, j] = f["jet"][k][s:e]
          cond[pos:pos+e-s, 4] = (codes[pos:pos+e-s, :, 0] >= 0).sum(1)
      pos += e-s
    return codes, cond

  def _pack(self, codes, cond):
    codes = torch.from_numpy(codes.astype(np.int64)) #embedding lookups need long
    if self.add_conditional:
      return [codes, torch.from_numpy(np.ascontiguousarray(cond))]
    return codes

  @property
  def cond(self):
    """the full [n_jets, 5] conditioning (20 bytes per jet, so fine to hold), e.g. for model.fit_cond_normalisation"""
    if self._cond_cache is None:
      if self.has_cond:
        with h5py.File(self.file_path, "r") as f:
          self._cond_cache = np.asarray(f["cond"][:self.n_jets], dtype=np.float32)
      else:
        self._cond_cache = np.concatenate([c for _, c in self.stream_all(need_cond=True)])
    return self._cond_cache

def get_code_loaders(args):
    #The tokenizer fixes the input format of the codes (needed for the validation loader and format_codes)
    _, tok_args = models_vqvae.load_tokenizer_checkpoint(helpers.config_get(args,"tokenizer"), "cpu")
    if args.input_format!=tok_args["input_format"]:
        print(f"Setting --input_format {tok_args['input_format']} (from the tokenizer, was {args.input_format})", flush=True)
        args.input_format=tok_args["input_format"]
        if args.input_format=="4vec" or args.input_format=="relvec": #NOTE preprocessing is applied in the lunddata script
            args.preprocess="logstd"

    add_conditional=(args.conditional or args.input_format=="relvec") #same rule as format_codes
    common=dict(add_conditional=add_conditional, batch_size=args.batch_size, num_workers=args.num_workers,
                block_size=helpers.config_get(args, "block_size", None),
                blocks_per_buffer=helpers.config_get(args, "blocks_per_buffer", 32) or 32,
                max_jets=helpers.config_get(args, "max_jets", None))
    train_dataset = codes_stream_dataset(args.train_codes, shuffle=args.shuffle, drop_last=args.shuffle, **common)
    test_dataset = codes_stream_dataset(args.test_codes, shuffle=False, drop_last=False, **common)

    #Number of constituents comes from the codes; the validation loader must use the same
    n_const=train_dataset.n_const
    if args.num_constituents!=n_const:
        print(f"Setting --num-constituents {n_const} (from the codes file, was {args.num_constituents})", flush=True)
        args.num_constituents=n_const

    #batch_size=None: the datasets already yield whole batches
    use_cuda = str(helpers.config_get(args, "device", "") or "").startswith("cuda") or torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=args.num_workers, pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=None, num_workers=args.num_workers, pin_memory=use_cuda)
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
    """
    New model from the run arguments (argparse namespace or dict)
    model.config stores the tokenizer config as well, so a checkpoint is self-contained.
    """
    architecture = helpers.config_get(args_or_dict, "architecture")
    if architecture=="VQGPT":
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
    raise ValueError(f"Unknown binned architecture {architecture}")
 
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
    parser.add_argument("--architecture","-a",type=str, default="VQGPT", choices=["VQGPT"], help="Architecture to run")
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
