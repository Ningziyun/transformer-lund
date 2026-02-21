#!/usr/bin/env python3

import numpy as np
import os,sys
import ROOT
import math
from tqdm import tqdm
#from argparse import ArgumentParser
import argparse  # CLI options

from helpers_train import *

import torch
from torchinfo import summary
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
torch.multiprocessing.set_sharing_strategy("file_system")
from torchviz import make_dot

import matplotlib.pyplot as plt

# -----------------------------
# Plotting utility (unchanged)
# -----------------------------
def quickLundPlot(inputs, labels=["original","generated","predicted"], epoch_losses=None, out_dir=""):

  linestyles=["-","--","-.",":"]

  Ndim=inputs[0].shape[1]
  Nin=len(inputs)

  mins=np.zeros(Ndim)
  maxs=np.zeros(Ndim)
  for ii in range(Ndim):
    for jj in range(Nin):
      mins[ii]=min(mins[ii],np.min(inputs[jj][:,ii]))
      maxs[ii]=max(maxs[ii],np.max(inputs[jj][:,ii]))

  #2d plots
  if Ndim>=2:
    fig, axs = plt.subplots(Nin,1,figsize=(8.0,8.0))
    for jj in range(Nin):
      pos=axs[jj].hist2d(inputs[jj][:,0],inputs[jj][:,1],range=[[mins[0],maxs[0]],[mins[1],maxs[1]]],bins=[20,20],cmap="Blues", norm="log")
      axs[jj].set_title(labels[jj])          # jj=0 -> original, jj=1 -> generated
      axs[jj].set_ylabel("log(kt)")
      if jj == Nin - 1:
        axs[jj].set_xlabel("log(1/deltaR)")
      else:
        axs[jj].set_xlabel("")
    fig.colorbar(pos[3],ax=axs)

    name="lund"
    fig.savefig(os.path.join(out_dir, name+".png"))
    fig.savefig(os.path.join(out_dir, name+".pdf"))
    plt.close(fig)

  #1D plots
  fig, axs = plt.subplots(Ndim,1,figsize=(8.0,8.0))
  if Ndim==1: axs=[axs]
  for ii in range(Ndim):
    for jj in range(Nin):
      axs[ii].hist(inputs[jj][:,ii],bins=20,range=[mins[ii],maxs[ii]],histtype="step",density=True,linestyle=linestyles[jj],label=labels[jj])
      axs[jj].set_title(labels[jj])          # jj=0 -> original, jj=1 -> generated
      axs[jj].set_ylabel("log(kt)")
      if jj == Nin - 1:
        axs[jj].set_xlabel("log(1/deltaR)")
      else:
        axs[jj].set_xlabel("")
    if ii==0: axs[0].legend()

  name="projection"
  fig.savefig(os.path.join(out_dir, name+".png"))
  fig.savefig(os.path.join(out_dir, name+".pdf"))
  plt.close(fig)

  #loss vs epoch plot if provided (Optional)
  if epoch_losses is not None and len(epoch_losses) > 0:
    fig = plt.figure(figsize=(6.0, 4.0))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.grid(True)
    fig.savefig(os.path.join(out_dir, "loss_vs_epoch.png"))
    fig.savefig(os.path.join(out_dir, "loss_vs_epoch.pdf"))
    plt.close(fig)


# -----------------------------
# Dataset (unchanged)
# -----------------------------
class input_dataset(torch.utils.data.Dataset):
  def __init__(self, file_path, NConstituents=7, add_stop=False, standardize=False):
    super(input_dataset, self).__init__()
    self.data=torch.tensor([])
    self.add_stop=add_stop

    Njets=-1
    df = pd.read_hdf(file_path, "raw", stop=Njets)

    cols=list(df)
    drcols=[col for col in cols if "deltaR" in col]
    ktcols=[col for col in cols if "kt" in col]
    drs=df[drcols].to_numpy()
    kts=df[ktcols].to_numpy()
    self.DR=drs[:,:NConstituents]
    self.kt=kts[:,:NConstituents]

    #Check if next is -1 and padd last const to True
    if add_stop:
      self.stop=self.DR[:,1:] == -1
      self.stop = np.concatenate([self.stop,  np.ones((self.stop.shape[0], 1), dtype=bool)],axis=1)

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
    return self.data

  def __len__(self):
    return len(self.DR)

def get_loaders(train_file, val_file, batch_size=256, num_workers=1, shuffle=True):
  train_dataset = input_dataset(train_file)
  train_loader = DataLoader( train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,)
  test_dataset = input_dataset(val_file)
  test_loader = DataLoader( test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,)
  return train_loader,test_loader


# -----------------------------
# Baseline models (unchanged)
# -----------------------------
class test_modelNN(nn.Module):
  def __init__(self, dim1, dim2):
      super(test_modelNN, self).__init__()

      self.dim1=dim1-1
      self.dim2=dim2

      self.fc1 = nn.Linear(self.dim1*self.dim2, 128)
      self.act1=nn.ReLU()
      self.fc2 = nn.Linear(128,128)
      self.act2=nn.ReLU()
      self.fc3 = nn.Linear(128, self.dim1*self.dim2)

  def forward(self, x):
      x=torch.flatten(x,start_dim=1)
      x = self.act1(self.fc1(x))
      x = self.act2(self.fc2(x))
      x = self.fc3(x)
      x = torch.reshape(x,[x.shape[0],self.dim1,self.dim2])
      return x


class test_model(nn.Module):
  def __init__(self, input_dim, embed_dim=256, num_heads=1, num_layers=2, ff_dim=128):
      super(test_model, self).__init__()

      self.input_dim=input_dim
      self.embed_dim=embed_dim
      self.ff_dim=ff_dim
      self.num_heads=num_heads
      self.num_layers=num_layers

      self.register_buffer("pos_encoding", self._build_pos_encoding(10, self.embed_dim))

      #Add the embedding layer
      self.embed=nn.Linear(input_dim,self.embed_dim)

      #specify the transformer block and number of layers
      encoder_layer=nn.TransformerEncoderLayer(
        d_model=self.embed_dim, nhead=self.num_heads, dim_feedforward=self.ff_dim,
        dropout=0.1, batch_first=True
      )
      self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

      #Now de-embed back to original output
      self.deembed = nn.Linear(self.embed_dim, self.input_dim)

  def forward(self, x):
      seq_len = x.shape[1] # (batch, seq_len, feature_dim)

      # Embed the N-dim vector into the embedded space
      x=self.embed(x) # (batch, seq_len, embed_dim)

      # Causal mask prevents looking ahead
      mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)

      encoded = self.encoder(x, mask=mask)
      return self.deembed(encoded)  # (batch, seq_len, feature_dim)

  def _build_pos_encoding(self, max_len, d_model):
        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        div_term = torch.exp(-torch.log(torch.tensor(10000.0)) * i / d_model)
        pos_encode = torch.zeros(max_len, d_model)
        pos_encode[:, 0::2] = torch.sin(pos * div_term)
        pos_encode[:, 1::2] = torch.cos(pos * div_term)
        return pos_encode.unsqueeze(0)

  @torch.no_grad()
  def generate(self, x_init, steps):
      seq = x_init.clone()
      for ii in range(steps):
          pred = self.forward(seq)
          next_pred = pred[:, -1:, :]
          seq = torch.cat([seq, next_pred], dim=1)
      return seq


##########################################
# MDN model (unchanged)
##########################################
class test_modelMDN(test_model):
  def __init__(self, input_dim, n_mix=25, embed_dim=128, num_heads=1, num_layers=2, ff_dim=128):
      super(test_modelMDN, self).__init__(input_dim, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim)

      self.n_mix=n_mix
      self.deembed = nn.Linear(self.embed_dim, self.n_mix*(1+input_dim+input_dim))

  def forward(self, x):
      encoded=super().forward(x)
      encoded=encoded.view(encoded.shape[0],encoded.shape[1],self.n_mix,(1+self.input_dim+self.input_dim))

      alpha=encoded[:,:,:,0]
      mu=encoded[:,:,:,1:self.input_dim+1]
      sigma=encoded[:,:,:,self.input_dim+1:]

      alpha = nn.functional.softmax(alpha, dim=-1)
      sigma=sigma.clamp(min=0.001)

      return torch.cat([alpha.unsqueeze(-1),mu,sigma],dim=-1)

  @torch.no_grad()
  def generate(self, x_init, steps):
      seq = x_init.clone()
      ninputs=x_init.shape[-1]
      batch_idx = torch.arange(x_init.shape[0])

      for ii in range(steps):
          pred = self.forward(seq)

          alpha=pred[:,-1,:,0]
          mu=pred[:,-1,:, 1:ninputs+1]
          sig2=pred[:,-1,:, ninputs+1:]

          if ii==0:
            print("alpha",alpha[0])
            print("mu",mu[0])
            print("sig2",sig2[0])

          comp = torch.multinomial(alpha, 1).squeeze(-1)
          loc=mu[batch_idx,comp,:]
          covmatrix = torch.diag_embed(sig2[batch_idx,comp,:]**2)
          dist = MultivariateNormal(loc,covmatrix)
          next_pred=dist.sample().unsqueeze(dim=1)
          seq = torch.cat([seq, next_pred], dim=1)
      return seq


def mdn_loss(inputs, targets, valid_mask=None):

    ninputs=targets.shape[-1]

    alpha=inputs[..., 0]
    mu=inputs[..., 1:ninputs+1]
    sig2=inputs[..., ninputs+1:]

    targets = targets.unsqueeze(2)

    Z_term = torch.sum( ((targets - mu)**2 / (2*sig2)), dim=-1)
    sig_term = 0.5*torch.sum(sig2+math.log(2*math.pi), dim=-1)
    alpha_term=torch.log(alpha)
    log_prob = torch.logsumexp(alpha_term - Z_term - sig_term, dim=-1)

    nll = -log_prob
    if valid_mask is not None:
        nll = nll[valid_mask]
    return nll.mean()


# ============================================================
# CNF MODE (Conditional Normalizing Flow) - NEW
# Notes:
# - This is a "conditional flow" over the next-step vector y in R^D.
# - We implement a light RealNVP-style coupling flow (commonly used as CNF in HEP workflows).
# - It conditions on the transformer's context vector at each time step.
# - Training uses exact log-likelihood, generation samples from the learned conditional distribution.
# ============================================================

class _CondCouplingNet(nn.Module):
  def __init__(self, in_dim, context_dim, hidden=128):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_dim + context_dim, hidden),
      nn.ReLU(),
      nn.Linear(hidden, hidden),
      nn.ReLU(),
      nn.Linear(hidden, 2)  # output: [s, t] for 1 transformed dimension (D=2 case)
    )

  def forward(self, x_part, context):
    # x_part: (B, 1), context: (B, C)
    h = torch.cat([x_part, context], dim=-1)
    out = self.net(h)
    s = out[:, :1]
    t = out[:, 1:2]
    return s, t


class _CondRealNVP2D(nn.Module):
  """
  Two-layer conditional RealNVP for 2D vectors.
  Masks alternate which dimension is transformed.
  """
  def __init__(self, context_dim, hidden=128, s_scale=1.5):
    super().__init__()
    self.s_scale = s_scale
    # layer 1: x0 stays, x1 transforms
    self.net10 = _CondCouplingNet(in_dim=1, context_dim=context_dim, hidden=hidden)
    # layer 2: x1 stays, x0 transforms
    self.net01 = _CondCouplingNet(in_dim=1, context_dim=context_dim, hidden=hidden)

  def _squash_s(self, s):
    # Keep scales bounded for stability
    return torch.tanh(s) * self.s_scale

  def forward(self, x, context):
    """
    Map data x -> latent z, and compute log|det J|.
    x: (B, 2), context: (B, C)
    """
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]

    # layer 1 (mask [1,0]): transform x1 conditioned on x0
    s1, t1 = self.net10(x0, context)
    s1 = self._squash_s(s1)
    z1 = x1 * torch.exp(s1) + t1
    logdet1 = s1.squeeze(-1)

    # layer 2 (mask [0,1]): transform x0 conditioned on z1
    s2, t2 = self.net01(z1, context)
    s2 = self._squash_s(s2)
    z0 = x0 * torch.exp(s2) + t2
    logdet2 = s2.squeeze(-1)

    z = torch.cat([z0, z1], dim=-1)
    logdet = logdet1 + logdet2
    return z, logdet

  def inverse(self, z, context):
    """
    Map latent z -> data x, and compute log|det J^{-1}|.
    z: (B, 2), context: (B, C)
    """
    z0 = z[:, 0:1]
    z1 = z[:, 1:2]

    # invert layer 2 first
    s2, t2 = self.net01(z1, context)
    s2 = self._squash_s(s2)
    x0 = (z0 - t2) * torch.exp(-s2)
    inv_logdet2 = (-s2).squeeze(-1)

    # invert layer 1
    s1, t1 = self.net10(x0, context)
    s1 = self._squash_s(s1)
    x1 = (z1 - t1) * torch.exp(-s1)
    inv_logdet1 = (-s1).squeeze(-1)

    x = torch.cat([x0, x1], dim=-1)
    inv_logdet = inv_logdet1 + inv_logdet2
    return x, inv_logdet

  def log_prob(self, x, context):
    """
    Exact conditional log-prob under base N(0, I) plus change-of-variables.
    """
    z, logdet = self.forward(x, context)
    # Standard normal log prob
    log_base = -0.5 * (z**2 + math.log(2*math.pi)).sum(dim=-1)
    return log_base + logdet

  def sample(self, context):
    """
    Sample x ~ p(x | context).
    context: (B, C)
    """
    B = context.shape[0]
    z = torch.randn(B, 2, device=context.device)
    x, _ = self.inverse(z, context)
    return x


class test_modelCNF(test_model):
  """
  Transformer backbone + conditional flow head.

  forward(x) returns a context tensor of shape (B, L, C),
  where C = embed_dim. The flow uses that context to model p(y_t | x_{<=t}).
  """
  def __init__(self, input_dim, embed_dim=256, num_heads=1, num_layers=2, ff_dim=128, flow_hidden=128):
    super(test_modelCNF, self).__init__(input_dim, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim)

    # We will NOT use deembed for CNF; we keep it but won't call it.
    self.flow = _CondRealNVP2D(context_dim=self.embed_dim, hidden=flow_hidden)

  def forward(self, x):
    # Build transformer context (B, L, embed_dim) with causal mask
    seq_len = x.shape[1]
    h = self.embed(x)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(h.device)
    context = self.encoder(h, mask=mask)
    return context

  @torch.no_grad()
  def generate(self, x_init, steps):
    seq = x_init.clone()
    for ii in range(steps):
      context = self.forward(seq)              # (B, cur_len, C)
      c_last = context[:, -1, :]              # (B, C)
      next_x = self.flow.sample(c_last)       # (B, 2)
      seq = torch.cat([seq, next_x.unsqueeze(1)], dim=1)
    return seq


def cnf_loss(context, targets, valid_mask=None, flow=None):
  """
  Negative log-likelihood for conditional flow.

  context: (B, L, C) from transformer
  targets: (B, L, D) next-step targets
  flow: the conditional flow module
  """
  # Flatten (B,L,*) -> (B*L,*)
  B, L, C = context.shape
  D = targets.shape[-1]

  ctx = context.reshape(B*L, C)
  y = targets.reshape(B*L, D)

  logp = flow.log_prob(y, ctx)  # (B*L,)

  nll = -logp
  if valid_mask is not None:
    vm = valid_mask.reshape(B*L)
    nll = nll[vm]
  return nll.mean()


# -----------------------------
# Quantile model (unchanged; not used)
# -----------------------------
class test_modelQuantile(test_model):
  def __init__(self, input_dim, embed_dim=128, num_heads=1, num_layers=2, ff_dim=128):
      super(test_model3, self).__init__(input_dim, embed_dim=128, num_heads=1, num_layers=2, ff_dim=128)

  def forward(self, x):
      seq_len = x.shape[1]
      x=self.embed(x)
      mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
      encoded = self.encoder(x)
      return self.deembed(encoded)

def quantile_loss(pred, target, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
      e = target - pred[:, :, i]
      losses.append(torch.max(q*e, (q-1)*e))
    return torch.mean(torch.stack(losses, dim=0))


# -----------------------------
# CLI (CNF flags added)
# -----------------------------
def parse_input():
    """Parse_Input args. Defaults match the current hard-coded values."""
    p = argparse.ArgumentParser(description="Train transformer/MDN on Lund data")

    # data / io
    p.add_argument("--train-file", default="inputFiles/discretized/qcd_lund_cut_train.h5", help="Path to training .h5")
    p.add_argument("--val-file", default="inputFiles/discretized/qcd_lund_cut_val.h5", help="Path to validation .h5 (unused yet)")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p.add_argument("--num-workers", type=int, default=1, help="DataLoader workers")
    p.add_argument("--shuffle", action="store_true", default=True, help="Shuffle training loader (default: True)")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable shuffle")
    p.add_argument("--num-features", dest="num_features", type=int, default=2,
                   help="Feature dimension per constituent (default: 2 = [deltaR, kt])")
    p.add_argument("--num-bins", dest="num_bins", type=int, nargs="+", default=[20, 20],
                   help="Binning spec (unused currently; default matches plotting bins)")

    # training
    p.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    p.add_argument("--seed", type=int, default=0, help="Random seed (overrides helpers_train default if you want)")

    # model switches
    p.add_argument("--mdn", action="store_true", default=True, help="Use MDN head (default: True)")
    p.add_argument("--no-mdn", dest="mdn", action="store_false", help="Disable MDN, use regression head")
    p.add_argument("--mixed-loss", action="store_true", default=False, help="Use mixed loss (default: False)")

    # CNF mode (new)
    p.add_argument("--cnf", action="store_true", default=False,
                   help="Use CNF (conditional normalizing flow) head (default: False)")
    p.add_argument("--no-cnf", dest="cnf", action="store_false",
                   help="Disable CNF head")

    # transformer hyperparams
    p.add_argument("--embed-dim", type=int, default=256, help="Transformer embedding dim")
    p.add_argument("--num-heads", type=int, default=1, help="Transformer num heads")
    p.add_argument("--num-layers", type=int, default=2, help="Transformer num layers")
    p.add_argument("--ff-dim", type=int, default=128, help="Transformer feedforward dim")

    # mdn hyperparams
    p.add_argument("--n-mix", type=int, default=25, help="Number of MDN mixtures")

    # cnf hyperparams (new)
    p.add_argument("--flow-hidden", type=int, default=128, help="Hidden size for CNF coupling nets")

    # misc
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="Force device; default auto")

    # logging / checkpointing
    p.add_argument("--log-dir", dest="log_dir", type=str, default="models/test",help="Logging directory")
    p.add_argument("--contin", action="store_true", default=False,help="Continue training from a saved model")
    p.add_argument("--model-path", dest="model_path", type=str, default="",help="Path to model/log_dir to load when --contin is set")
    return p.parse_args()


# -----------------------------
# Main (only CNF branches added)
# -----------------------------
if __name__ == "__main__":
    args = parse_input()
    save_arguments(args)
    print(f"Logging to {args.log_dir}")
    set_seeds(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device
    print(f"Running on device: {device}")

    num_features = args.num_features
    num_bins = tuple(args.num_bins)

    # load and preprocess data
    print(f"Loading training set")

    train_loader, test_loader = get_loaders(
      train_file=args.train_file, val_file=args.val_file,
      batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle
    )

    X=next(iter(train_loader))
    print("Input shape,",X.shape)

    # Mode switches
    doCNF = args.cnf
    doMDN = args.mdn
    doMixedLoss = args.mixed_loss

    # If CNF is enabled, we ignore MDN/regression heads
    if doCNF:
      doMDN = False
      doMixedLoss = False

    # construct model
    if args.contin:
        model = load_model(log_dir=args.model_path)
        print("Loaded model")
    else:
        #model=test_modelNN(X.shape[1],X.shape[2])
        if doCNF:
          model = test_modelCNF(
            input_dim=X.shape[2],
            embed_dim=args.embed_dim, num_heads=args.num_heads,
            num_layers=args.num_layers, ff_dim=args.ff_dim,
            flow_hidden=args.flow_hidden
          )
        elif doMDN:
          model = test_modelMDN(
            input_dim=X.shape[2], n_mix=args.n_mix,
            embed_dim=args.embed_dim, num_heads=args.num_heads,
            num_layers=args.num_layers, ff_dim=args.ff_dim
          )
        else:
          model = test_model(
            input_dim=X.shape[2],
            embed_dim=args.embed_dim, num_heads=args.num_heads,
            num_layers=args.num_layers, ff_dim=args.ff_dim
          )

    summary(model,input_data=[X[:,:-1,:]], col_names=["input_size", "output_size", "num_params","params_percent","mult_adds","trainable"])

    # For CNF, model(X[:,:-1,:]) returns context, not feature predictions
    print("Output shape,",model(X[:,:-1,:]).shape)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss functions
    if doCNF:
      loss_fn = None  # we call cnf_loss(...) directly
    elif not doMDN:
      loss_fn = nn.MSELoss(reduction='none')
      if doMixedLoss:
        loss_fn2 = nn.BCEWithLogitsLoss(reduction='none')
        sigmoid=nn.Sigmoid()
    else:
      loss_fn=mdn_loss

    best_loss=1e6
    patience_counter=0
    patience = args.patience
    loss_test=[]
    loss_train=[]
    epochs = args.epochs
    train_epoch_losses = []

    for t in range(epochs):
      epoch_loss_sum = 0.0
      epoch_n_batches = 0

      for batch, X in enumerate(train_loader):
          X = X.to(device)
          optimizer.zero_grad()

          inputs = X[:, :-1, :]
          targets = X[:, 1:, :]

          # Mask out padding tokens: both deltaR and kt are -1
          valid_mask = ~((targets[:, :, 0] == -1) & (targets[:, :, 1] == -1))

          if doCNF:
            context = model(inputs)  # (B, L, C)
            loss = cnf_loss(context, targets, valid_mask=valid_mask, flow=model.flow)
          else:
            pred = model(inputs)

            if not doMDN:
              if doMixedLoss:
                lambd=1
                loss= loss_fn(pred[:,:,:-1],targets[:,:,:-1]).mean(dim=-1)+lambd*loss_fn2(pred[:,:,-1],targets[:,:,-1])
              else:
                loss = loss_fn(pred, targets)
              loss = loss[valid_mask].mean()
            else:
              loss = loss_fn(pred, targets, valid_mask=valid_mask)
              loss = loss.mean()

          epoch_loss_sum += loss.item()
          epoch_n_batches += 1

          loss.backward()
          optimizer.step()

          if batch % 100 == 0:
            print(f"batch: {batch} loss:{loss.item()}")

      # Keep original mixed-loss postprocessing (only relevant if not CNF)
      if (not doCNF) and doMixedLoss:
        pred[:,:,-1]=sigmoid(pred[:,:,-1])

      print(f"epoch: {t} loss={loss.item()}")

      epoch_avg_loss = epoch_loss_sum / max(epoch_n_batches, 1)
      train_epoch_losses.append(epoch_avg_loss)

      loss_train.append(loss.item())
      if loss_train[-1]<best_loss:
        best_loss=loss_train[-1]
        patience_counter=0
      else:
        patience_counter+=1
        if patience_counter>=patience:
          print("Early stoping")
          break

    with torch.no_grad():
      original=torch.empty([0,X.shape[-1]])
      generated=torch.empty([0,X.shape[-1]])
      predicted=torch.empty([0,X.shape[-1]])
      device="cpu"

      for batch, X in enumerate(train_loader):
        if batch>10: break
        X = X.to(device)
        start=X[:,0,:].unsqueeze(1)
        model.to(device)

        # For CNF we don't have "pred" in the same sense; keep the rest unchanged
        pred= model(X) if (not doCNF) else model(X[:,:-1,:])

        generated_seq = model.generate(start, steps=X.shape[1]-1)

        if (not doCNF) and doMixedLoss:
          pred[:,:,-1]=sigmoid(pred[:,:,-1])
          generated_seq[:,:,-1]=sigmoid(generated_seq[:,:,-1])
          generated_seq[:,0,-1]=X[:,0,-1]

        if batch==0:
          print("Input example")
          print(X[0])
          print("Generate example")
          print(generated_seq[0])

        original=torch.cat([original,X.flatten(0,1)])
        generated=torch.cat([generated,generated_seq.flatten(0,1)])

      quickLundPlot([original.numpy(),generated.numpy()],epoch_losses=train_epoch_losses,out_dir=args.log_dir)
