import os,sys
import time
import numpy as np
import pandas as pd
import ROOT
import math
import matplotlib.pyplot as plt

from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torchinfo import summary
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
torch.multiprocessing.set_sharing_strategy("file_system")
from torchviz import make_dot

class ktdr_dataset(torch.utils.data.Dataset):
  def __init__(self, file_path, NConstituents=50, add_stop=False, standardize=False):
    super(ktdr_dataset, self).__init__()
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

class constit_dataset(torch.utils.data.Dataset):
  def __init__(self, file_path, NConstituents=50, add_stop=False, standardize=False):
    super(constit_dataset, self).__init__()
    self.data=torch.tensor([])
    self.add_stop=add_stop

    Njets=-1
    df = pd.read_hdf(file_path, "table", stop=Njets)
    df = df.loc[df['is_signal_new'] == 0]

    cols=list(df)
    Ecols=[col for col in cols if "E_" in col]
    Pxcols=[col for col in cols if "PX_" in col]
    Pycols=[col for col in cols if "PY_" in col]
    Pzcols=[col for col in cols if "PZ_" in col]
    es=df[Ecols].to_numpy()
    pxs=df[Pxcols].to_numpy()
    pys=df[Pycols].to_numpy()
    pzs=df[Pzcols].to_numpy()
    self.E=es[:,:NConstituents]
    self.px=pxs[:,:NConstituents]
    self.py=pys[:,:NConstituents]
    self.pz=pzs[:,:NConstituents]

    #Check if next is -1 and padd last const to True
    if add_stop:
      self.stop=self.E[:,1:] == 0
      self.stop = np.concatenate([self.stop,  np.ones((self.stop.shape[0], 1), dtype=bool)],axis=1)

    '''
    if standardize:
      dr_mean,dr_std=1.782,1.084
      kt_mean,kt_std=1.397,1.117
      self.DR=(self.DR-dr_mean)/dr_std
      self.kt=(self.kt-kt_mean)/kt_std
    '''

  def __getitem__(self, index):
    inputs=np.array([self.E[index],self.px[index],self.px[index],self.pz[index]])
    if self.add_stop:
      inputs=np.concatenate([inputs,[self.stop[index]]],axis=0)
    self.data=torch.transpose(torch.tensor(inputs),0,1)

    return self.data

  def __len__(self):
    return len(self.E)

def get_loaders(input_format="ktdr",batch_size=256, num_workers=1, shuffle=True):

  if input_format=="ktdr":
    train_dataset = ktdr_dataset("inputFiles/discretized/qcd_lund_cut_lundTree_kt_deltaR_train.h5")
    train_loader = DataLoader( train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,)
    test_dataset = ktdr_dataset("inputFiles/discretized/qcd_lund_cut_lundTree_kt_deltaR_val.h5")
    test_loader = DataLoader( test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,)

  elif input_format=="4vec":
    train_dataset = constit_dataset("inputFiles/top_benchmark/train.h5")
    train_loader = DataLoader( train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,)
    test_dataset = constit_dataset("inputFiles/top_benchmark/test.h5")
    test_loader = DataLoader( test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,)
  return train_loader,test_loader

class model_DNN(nn.Module):
  def __init__(self, dim1, dim2):
      super(model_DNN, self).__init__()

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

class model_transformer(nn.Module):
  def __init__(self, input_dim, embed_dim=256, num_heads=1, num_layers=2, ff_dim=128):
      super(model_transformer, self).__init__()

      self.input_dim=input_dim
      self.embed_dim=embed_dim
      self.ff_dim=ff_dim
      self.num_heads=num_heads
      self.num_layers=num_layers

      self.register_buffer("pos_encoding", self._build_pos_encoding(10, self.embed_dim)) #makes a self.pos_encoding which is registered as buffer on correct device, saves a .to(x.device) call later

      #Add the embedding layer
      self.embed=nn.Linear(input_dim,self.embed_dim)

      #specify the transformer block and number of layers
      encoder_layer=nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=0.1, batch_first=True)
      self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

      #Now de-embed back to original output
      self.deembed = nn.Linear(self.embed_dim, self.input_dim)

  def forward(self, x):

      seq_len = x.shape[1] # (batch, seq_len, feature_dim)

      # Embed the N-dim vector into the embedded space
      x=self.embed(x) # (batch, seq_len, embed_dim)
      #x=x+self.pos_encoding[:, :seq_len, :]

      # Causal mask prevents looking ahead
      mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)

      encoded = self.encoder(x, mask=mask)
      #encoded = self.encoder(x)
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
          pred = self.forward(seq) #get next element prediction, gives you N prediction for N inputs
          next_pred = pred[:, -1:, :]  # Just take the last one
          seq = torch.cat([seq, next_pred], dim=1) #append it
      return seq


class model_transformer_MDN(model_transformer):
  def __init__(self, input_dim, n_mix=25, embed_dim=128, num_heads=1, num_layers=2, ff_dim=128):
      super(model_transformer_MDN, self).__init__(input_dim, embed_dim=128, num_heads=1, num_layers=2, ff_dim=128)

      self.n_mix=n_mix

      self.deembed = nn.Linear(self.embed_dim, self.n_mix*(1+input_dim+input_dim))

  def forward(self, x):

      #Get the usual network result
      encoded=super().forward(x) #[Nbatch,Nconst,Nmix*(1+2*Ninput)]

      # split into mixture components
      encoded=encoded.view(encoded.shape[0],encoded.shape[1],self.n_mix,(1+self.input_dim+self.input_dim)) #[Nbatch,Nconst,Nmix,(1+2*Ninput)]

      alpha=encoded[:,:,:,0] #[batch,Nconst,Nmix]
      mu=encoded[:,:,:,1:self.input_dim+1] #[batch,Nconst,Nmix,Ninput]
      sigma=encoded[:,:,:,self.input_dim+1:] #[batch,Nconst,Nmix,Ninput]

      # constraints, don't do in-line replacements of tensors as can mess with gradients
      alpha = nn.functional.softmax(alpha, dim=-1) #weights need to be normalized
      sigma=sigma.clamp(min=0.001) #make positive

      return torch.cat([alpha.unsqueeze(-1),mu,sigma],dim=-1)
      #return alpha, mu, sigma

  @torch.no_grad()
  def generate(self, x_init, steps):
      seq = x_init.clone()
      ninputs=x_init.shape[-1]
      batch_idx = torch.arange(x_init.shape[0]) #For some smoother slicing later

      for ii in range(steps):
          pred = self.forward(seq) #get the alpha,mu,sigma values

          #Take the last nconst and get the components
          alpha=pred[:,-1,:,0] # [Nbatch, Nmix]
          mu=pred[:,-1,:, 1:ninputs+1] #[Nbatch,Nmix,Ninput]
          sig2=pred[:,-1,:, ninputs+1:] #[Nbatch,Nmix,Ninput]

          if ii==0:
            print("alpha",alpha[0])
            print("mu",mu[0])
            print("sig2",sig2[0])

          # sample component index, grab the multi-nominal result, which returns the selected mix compoenent
          comp = torch.multinomial(alpha, 1).squeeze(-1)  # (B,)

          # Make the MVN distribution by getteing the mu and cov-matrix for this component and sample from it
          loc=mu[batch_idx,comp,:] #(Nbatch,Ninput)
          covmatrix = torch.diag_embed(sig2[batch_idx,comp,:]**2) # (Nbatch, Ninput, Ninput)
          dist = MultivariateNormal(loc,covmatrix)
          next_pred=dist.sample().unsqueeze(dim=1)
          seq = torch.cat([seq, next_pred], dim=1) #append it
      return seq

def mdn_loss(inputs, targets):

    ninputs=targets.shape[-1]

    alpha=inputs[..., 0] #[Nbatch,NConst,Nmix]
    mu=inputs[..., 1:ninputs+1] #target: [Nbatch,NConst,Nmix,Ninputs]
    sig2=inputs[..., ninputs+1:] #target: [Nbatch,NConst,Nmix,Ninputs]

    #target: [Nbatch,NConst,Ninputs]
    targets = targets.unsqueeze(2)  # target: [Nbatch,NConst,1,Ninputs]

    # central term, sum over the input vector dimension: (sum_{j=1}^{N_input} (x-mu_j)^2/2sigma_j^2)
    Z_term = torch.sum( ((targets - mu)**2 / (2*sig2)), dim=-1)  #[Nbatch,NConst,Nmix]

    # Norm term: sum_{j=1}^{N_input} 0.5*log(det|Sigma|)+N_input/2*log(2pi) #Assume diagonal and no const = 0.5*sum_{j=1}^{Ninput} sigma_j^2
    sig_term = 0.5*torch.sum(sig2+math.log(2*math.pi), dim=-1)  #[Nbatch,NConst,Nmix]
    #sig_term = 0.5*torch.sum(sig2, dim=-1)

    #the mixture term: log(alpha_i)
    alpha_term=torch.log(alpha)

    #Total log prob of the datapoint, sum over mixture: log(p_{sample})=log(sum_{i=1}^{N_mix} alpha,i*exp{-sig_term,i}*exp{-Z_term,i})
    #Make simpler and more stabler by doing the log-sum-exp: log(p_sample)=log(sum_{i=1}^{N_mix} exp{alpha_term,i - Z_term,i -exp{sig_term,i})
    log_prob = torch.logsumexp(alpha_term - Z_term - sig_term, dim=-1) #[Nbatch,NConst]

    # -log(p)= -log(prod {p_sample}) = -sum log(p_{sample})
    return -log_prob.mean() #Sum over all the training sample

class model_transformer_quantile(model_transformer):
  def __init__(self, input_dim, embed_dim=128, num_heads=1, num_layers=2, ff_dim=128):
      super(model_transformer_quantile, self).__init__(input_dim, embed_dim=128, num_heads=1, num_layers=2, ff_dim=128)

  def forward(self, x):
      seq_len = x.shape[1] # (batch, seq_len, feature_dim)

      # Embed the N-dim vector into the embedded space
      x=self.embed(x) # (batch, seq_len, embed_dim)
      #x=x+self.pos_encoding[:, :seq_len, :]

      # Causal mask prevents looking ahead
      mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)

      encoded = self.encoder(x)
      return self.deembed(encoded)  # (batch, seq_len, feature_dim)

def quantile_loss(pred, target, quantiles):
    # pred: [B, D, Q], target: [B, D]
    losses = []
    for i, q in enumerate(quantiles):
      e = target - pred[:, :, i]
      losses.append(torch.max(q*e, (q-1)*e))
    return torch.mean(torch.stack(losses, dim=0))

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


def save_opt_states(optimizer, scheduler, scaler, log_dir):
    torch.save(
        {
            "opt_state_dict": optimizer.state_dict(),
            "sched_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        os.path.join(log_dir, "opt_state_dict.pt"),
    )


def load_opt_states(optimizer, scheduler, scaler, log_dir):
    state_dicts = torch.load(os.path.join(log_dir, "opt_state_dict.pt"))
    optimizer.load_state_dict(state_dicts["opt_state_dict"])
    scheduler.load_state_dict(state_dicts["sched_state_dict"])
    scaler.load_state_dict(state_dicts["scaler_state_dict"])


def save_model(model, log_dir, name):
    torch.save(model, os.path.join(log_dir, f"model_{name}.pt"))

def load_model(model_path):
    model = torch.load(model_path)
    return model

def parse_input():
    parser = ArgumentParser()
    parser.add_argument(
        "--log_dir", type=str, default="models/test", help="Model directory"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="inputFiles/top_benchmark/train_qcd_30_bins.h5",
        help="Path to training data file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model file to load",
    )
    parser.add_argument(
        "--sample_file",
        type=str,
        default=None,
        help="Path to file for sampling. If none given, sample from model",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="the random seed for torch and numpy"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Training steps between logging"
    )
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        default=5000,
        help="Training steps between saving checkpoints",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    parser.add_argument(
        "--num_const", type=int, default=100, help="Number of constituents"
    )
    parser.add_argument(
        "--limit_const",
        action="store_true",
        help="Only use jets with at least num_const constituents",
    )
    parser.add_argument(
        "--num_events", type=int, default=-1, help="Number of events for training"
    )
    parser.add_argument(
        "--start_token",
        action="store_true",
        help="Whether to use a start particle (learn first particle as well)",
    )
    parser.add_argument(
        "--end_token",
        action="store_true",
        help="Whether to use a end particle (learn jet length as well)",
    )
    parser.add_argument(
        "--contin", action="store_true", help="Whether to continue training"
    )
    parser.add_argument(
        "--global_step", type=int, default=0, help="Starting point of step counter"
    )
    parser.add_argument(
        "--reverse", action="store_true", help="Whether to reverse pt order"
    )

    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=0.01, help="learning rate decay (linear)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.00001, help="weight decay"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cos",
        choices=["cos", "lin", "exp"],
        help="Type of learning rate scheduler to use (cos, lin, exp)."
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dim of the model"
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="Enable early stopping based on validation loss."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs with no sufficient improvement before stopping."
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="Minimum absolute improvement in validation loss to reset patience."
    )

    parser.add_argument("--doMDN", action="store_true", help="Run Mixture Density Network Variant")
    parser.add_argument("--doMixedLoss", action="store_true", help="Use a mixed loss")
    parser.add_argument("--input_format", type=str, choices=["ktdr","4vec"], help="What format of inputs we are using")

    args = parser.parse_args()
    return args

def save_arguments(args):
    tmp = args.log_dir
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

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def projection_plot(inputs,labels=["original","generated","predicted"],outdir="./Plots/"):

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  linestyles=["-","--","-.",":"]

  #Get ranges
  Ndim=inputs[0].shape[1]
  Nin=len(inputs)
  mins=np.zeros(Ndim)
  maxs=np.zeros(Ndim)
  for ii in range(Ndim):
    for jj in range(Nin):
      mins[ii]=min(mins[ii],np.min(inputs[jj][:,ii]))
      maxs[ii]=max(maxs[ii],np.max(inputs[jj][:,ii]))

  #make plot
  fig, axs = plt.subplots(Ndim,1,figsize=(8.0,8.0))
  if Ndim==1: axs=[axs]
  for ii in range(Ndim):
    for jj in range(Nin):
      axs[ii].hist(inputs[jj][:,ii],bins=20,range=[mins[ii],maxs[ii]],histtype="step",density=True,linestyle=linestyles[jj],label=labels[jj])
    if ii==0: axs[0].legend()

  name="projection"
  fig.savefig(outdir+name+".png")
  fig.savefig(outdir+name+".pdf")
  plt.close(fig)

def lund_plot(inputs,labels=["original","generated","predicted"],outdir="./Plots/"):

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  linestyles=["-","--","-.",":"]

  #Get ranges
  Ndim=inputs[0].shape[1]
  Nin=len(inputs)
  mins=np.zeros(Ndim)
  maxs=np.zeros(Ndim)
  for ii in range(Ndim):
    for jj in range(Nin):
      mins[ii]=min(mins[ii],np.min(inputs[jj][:,ii]))
      maxs[ii]=max(maxs[ii],np.max(inputs[jj][:,ii]))

  #Make plot
  if Ndim>=2:
    fig, axs = plt.subplots(Nin,1,figsize=(8.0,8.0))
    for jj in range(Nin):
      pos=axs[jj].hist2d(inputs[jj][:,0],inputs[jj][:,1],range=[[mins[0],maxs[0]],[mins[1],maxs[1]]],bins=[20,20],cmap="Blues", norm="log")
    fig.colorbar(pos[3],ax=axs)

    name="lund"
    fig.savefig(outdir+name+".png")
    fig.savefig(outdir+name+".pdf")
    plt.close(fig)

if __name__ == "__main__":
    pass
