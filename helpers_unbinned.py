import os,sys
import time
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", f"matplotlib-{os.environ.get('USER', 'user')}"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import numpy as np
import pandas as pd
import h5py
import math
import csv
import re
import textwrap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tqdm import tqdm
import argparse

import gc

import torch
from torchinfo import summary
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim.lr_scheduler import LambdaLR
torch.multiprocessing.set_sharing_strategy("file_system")
from torchviz import make_dot

import normflows as nf
from torchdiffeq import odeint_adjoint as odeint

import fastjet
import ljpHelpers

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

class model_autoregressive_transformer(nn.Module):
    #Auto-regressive trasnformer, learns next element prediction p(x_i|x_{<i}). During training takes x[0:-1] and learns to predict x[1:] via a transformer. For generation always needs a seed x[0], then can recursively generate the rest of the elements
  def __init__(self, input_dim, embed_dim=256, num_heads=1, num_layers=2, ff_dim=128):
      super(model_autoregressive_transformer, self).__init__()

      self.input_dim=input_dim
      self.embed_dim=embed_dim
      self.ff_dim=ff_dim
      self.num_heads=num_heads
      self.num_layers=num_layers

      #Add the embedding layer
      self.embed=nn.Linear(input_dim, self.embed_dim)

      #specify the transformer block and number of layers
      encoder_layer=nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=0.1, batch_first=True)
      self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

      #Now de-embed back to original output
      self.deembed = nn.Linear(self.embed_dim, self.input_dim)

  def forward(self, x):

      seq_len = x.shape[1] # (batch, seq_len, feature_dim)

      # Embed the N-dim vector into the embedded space
      x=self.embed(x) # (batch, seq_len, embed_dim)

      # Causal mask prevents looking ahead
      mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)

      # Take the x[0:-1] embedded and learn the embedded x[1:]
      encoded = self.encoder(x, mask=mask)  
      return self.deembed(encoded)  # (batch, seq_len, feature_dim)

  def mse_loss(self, pred, targets):
      loss_fn = nn.MSELoss(reduction='none')   # regression next-step prediction
      '''
      if args.mixed_loss:
        #loss_fn2 = nn.CrossEntropyLoss() #expects logits
        loss_fn2 = nn.BCEWithLogitsLoss(reduction='none') #expects logits
        sigmoid=nn.Sigmoid()
      '''

      return loss_fn(pred,targets)

  @torch.no_grad()
  def generate(self, out_dimensions):
      seq=torch.zeros(out_dimensions[0],1,out_dimensions[2])
      steps=out_dimensions[1]
      for ii in range(steps): #loop over length
          pred = self.forward(seq) #get next element prediction, gives you N prediction for N inputs
          next_pred = pred[:, -1:, :]  # Just take the last prediction which is new
          seq = torch.cat([seq, next_pred], dim=1) #append it to the sequence
      return seq[:,1:,:]

class model_autoregressive_transformer_MDN(model_autoregressive_transformer):
  #exactly like the previous auto-regressive model, but models the next prediction as a gaussian mixture model as opposed to exact value. Seems to avoid mode collapse
  def __init__(self, input_dim, n_mix=25, embed_dim=128, num_heads=1, num_layers=2, ff_dim=128):
      super(model_autoregressive_transformer_MDN, self).__init__(input_dim, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim)

      self.n_mix=n_mix

      #Final space is now a multi-D Gaussian mixture model: prod \alpha_i Gaus(\arrow\mu_i,\arrow\sigma_i) , where dimension=length of ouput (aka 4 for 4-vec)
      self.deembed = nn.Linear(self.embed_dim, self.n_mix*(1+input_dim+input_dim))

  def forward(self, x):

      #Get the usual network result, note we overloaded the original forward to give MDN values and not truly auto-regressive
      encoded=super().forward(x) #[Nbatch,Nconst,Nmix*(1+2*Ninput)]

      # split into mixture components: here alpha=norm (logits will softmax later), mu=center, sigma2=variance
      encoded=encoded.view(encoded.shape[0],encoded.shape[1],self.n_mix,(1+self.input_dim+self.input_dim)) #[Nbatch,Nconst,Nmix,(1+2*Ninput)]
      alpha=encoded[:,:,:,0] #[batch,Nconst,Nmix]
      mu=encoded[:,:,:,1:self.input_dim+1] #[batch,Nconst,Nmix,Ninput]
      sigma2=encoded[:,:,:,self.input_dim+1:] #[batch,Nconst,Nmix,Ninput]

      # constraints, don't do in-line replacements of tensors as can mess with gradients
      #alpha = nn.functional.softmax(alpha, dim=-1) #weights need to be normalized
      sigma2=sigma2.clamp(0.001, 10)

      assert torch.isfinite(alpha).all()
      assert torch.isfinite(mu).all()
      assert torch.isfinite(sigma2).all()

      return torch.cat([alpha.unsqueeze(-1),mu,sigma2],dim=-1)

  def nll_loss(self, inputs, targets, mask=None):
    ninputs=targets.shape[-1]

    alpha=inputs[..., 0] #[Nbatch,NConst,Nmix]
    mu=inputs[..., 1:ninputs+1] #target: [Nbatch,NConst,Nmix,Ninputs]
    sig2=inputs[..., ninputs+1:] #target: [Nbatch,NConst,Nmix,Ninputs]

    #target: [Nbatch,NConst,Ninputs]
    targets = targets.unsqueeze(2)  # target: [Nbatch,NConst,1,Ninputs]

    # central term, sum over the input vector dimension: (sum_{j=1}^{N_input} (x-mu_j)^2/2sigma_j^2)
    Z_term = torch.sum(((targets - mu)**2 / (2*sig2)), dim=-1)  #[Nbatch,NConst,Nmix]

    # Norm term: sum_{j=1}^{N_input} 0.5*log(det|covariance|)+N_input/2*log(2pi) #Assume diagonal and no const = 0.5*sum_{j=1}^{Ninput} sigma_j^2
    sig_term = 0.5*torch.sum(torch.log(sig2)+math.log(2*math.pi), dim=-1)  #[Nbatch,NConst,Nmix]

    #the mixture term: log(alpha_i), al
    #alpha_term=torch.log(alpha)
    alpha_term=F.log_softmax(alpha, dim=-1) #more stable to log_softmax

    #Total log prob of the datapoint, sum over mixture: log(p_{sample})=log(sum_{i=1}^{N_mix} alpha,i*exp{-sig_term,i}*exp{-Z_term,i})
    #Make simpler and more stabler by doing the log-sum-exp: log(p_sample)=log(sum_{i=1}^{N_mix} exp{alpha_term,i - Z_term,i -exp{sig_term,i})
    log_prob = torch.logsumexp(alpha_term - Z_term - sig_term, dim=-1) #[Nbatch,NConst]

    # -log(p)= -log(prod {p_sample}) = -sum log(p_{sample})
    if mask is not None:
      return -log_prob[mask].sum()  # mean over valid tokens only
    else:
      return -log_prob.sum() #Sum over all the training sample

  @torch.no_grad()
  def generate(self, out_dimensions):
      seq=torch.zeros(out_dimensions[0],1,out_dimensions[2])
      steps=out_dimensions[1]
      ninputs=out_dimensions[-1]
      batch_idx=torch.arange(out_dimensions[0]) #For some smoother slicing later

      for ii in range(steps):
          pred = self.forward(seq) #get the alpha,mu,sigma values

          #Take the last nconst and get the components
          alpha=F.softmax(pred[:,-1,:,0], dim=-1) # [Nbatch, Nmix]
          mu=pred[:,-1,:, 1:ninputs+1] #[Nbatch,Nmix,Ninput]
          sig2=pred[:,-1,:, ninputs+1:] #[Nbatch,Nmix,Ninput]

          # sample component index, grab the multi-nominal result, which returns the selected mix compoenent
          comp = torch.multinomial(alpha, 1).squeeze(-1)  # (B,)

          # Sample the whole MDN distribution by getting the mu and cov-matrix for this component and sample from it
          loc=mu[batch_idx,comp,:] #(Nbatch,Ninput)
          covmatrix = torch.diag_embed(sig2[batch_idx,comp,:]) # (Nbatch, Ninput, Ninput)
          dist = MultivariateNormal(loc,covmatrix)
          next_pred=dist.sample().unsqueeze(dim=1)
          seq = torch.cat([seq, next_pred], dim=1) #append it
      return seq[:,1:,:]

# =========================
# CNF (Continuous Normalizing Flow) components
# =========================
class VectorFieldNN(nn.Module):
  """
  Conditional vector field: dz/dt = f(z, t, c)
  z: [B, D]
  c: [B, C] (context from transformer)
  """
  def __init__(self, z_dim, c_dim, hidden_dim=128):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(z_dim + c_dim + 1, hidden_dim),  # +1 for time embedding (t)
      nn.SiLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, z_dim),
    )

  def forward(self, z, t, c=None):
    tt = t.expand(z.shape[0], 1) #should be on device
    if c==None:
        zct = torch.cat([z, tt], dim=-1)
    else:
        zct = torch.cat([z, c, tt], dim=-1)
    return self.net(zct)

def hutch_trace(f, z, eps):
  """
  Hutchinson trace estimator for divergence: tr(df/dt) approx E_p(epsilon){epsilon*df/ft*epilson} for gaussian vector noise epislone
  f: [B, D], x requires_grad=True
  returns: [B]
  """
  v = (f * eps).sum()
  grad = torch.autograd.grad(
        outputs=v,
        inputs=z,
        #grad_outputs=eps,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
  return (grad * eps).sum(dim=-1)

class CNFDynamics(nn.Module):
  """
  A minimal CNF using RK4 integration + Hutchinson trace estimator.
  This avoids external dependencies (torchdiffeq), and is enough to get a working CNF mode.
  """
  def __init__(self, x_dim, c_dim, hidden_dim=128, steps=100):
    super().__init__()
    self.x_dim = x_dim
    self.c_dim = c_dim
    self.hidden_dim = hidden_dim

    self.vf = VectorFieldNN(x_dim, c_dim, hidden_dim=hidden_dim)

    self.steps = steps
    self.register_buffer("time_grid", torch.linspace(0, 1, steps+1))

  def forward(self, x):
    """
    Compute log p(z | c) by integrating backward from x -> z (base)
    Evolution:  z_0                 = int_t_1^t_0 f(z, t)*dt      with z(t_1) = x
                log(p(x))-log(p(z)) = int_t_1^t_0 -tr( df/dt )*dt with log(p(x))-log(p(z(t_1)=0
    x: [B, D]
    c:  [B, C]
    """
    device, dtype = x.device, x.dtype
    t = self.time_grid.to(device=device, dtype=dtype)

    #Intital values: x=0 and prob difference is zero, re-use the eps for numerical speed-up
    x = x.requires_grad_(True)
    delta_logp = torch.zeros(x.shape[0], device=device, dtype=dtype)
    eps=torch.randn_like(x)

    # integrate backward
    dt=1/self.steps
    for k in range(len(t) - 1, 0, -1):
      k1 = self.vf(x, t[k])
      k2 = self.vf(x + 0.5 * dt * k1, t[k] + 0.5 * dt)
      k3 = self.vf(x + 0.5 * dt * k2, t[k] + 0.5 * dt)
      k4 = self.vf(x + dt * k3, t[k] + dt)
      x = x - dt * (k1 + 2*k2 + 2*k3 + k4)/6
      #x = x - dt * f  

      f = self.vf(x, t[k])
      div = hutch_trace(f, x, eps)  # divergence estimate
      delta_logp = delta_logp + dt * div  #log(p(x_{k_1}))= log(p(x_{k_1})) + tr(df/dx)*df
    z = x

    # standard normal base log prob
    logp0 = ( -0.5 * z**2 - 0.5 * torch.log( torch.tensor( 2.0 * torch.pi, device=z.device))).sum(dim=1)

    #want to minimize the nll_loss: log(p(x)) ~ log(p(z_0)) - delta_log(p) #note our delta_logp si flipped sign since inverse int direction before
    return logp0 - delta_logp

  @torch.no_grad()
  def generate(self, batch_size, c=None):
    """
    Sample x ~ p(z|c) by sampling z~N(0,I) and integrating forward.
    c: [B, C]
    returns x [B, D]
    """
    device = next(self.parameters()).device
    #dtype=c.dtype

    t = self.time_grid.to(device=device)
    z = torch.randn(batch_size, self.x_dim, device=device,)

    # integrate forward: x_{k+1} = x_k + dt * f(t_k, x_k, c)
    dt=1/self.steps
    for k in range(0, len(t) - 1):
      k1 = self.vf(z, t[k])
      k2 = self.vf(z + 0.5 * dt * k1, t[k] + 0.5 * dt)
      k3 = self.vf(z + 0.5 * dt * k2, t[k] + 0.5 * dt)
      k4 = self.vf(z + dt * k3, t[k] + dt)
      z = z + dt * ( k1 + 2*k2 + 2*k3 + k4) / 6.0

      #f = self.vf(z, t[k], c)
      #z = z + dt * f
    x=z

    return x

class model_CNF(nn.Module):
  """
  Transformer context encoder + CNF head for conditional density p(x_{t+1} | x_{<=t}).
  This produces a likelihood-based model (harder constraints require bounded transforms; CNF itself is on R^D).
  """
  def __init__(self, input_dim, embed_dim=256, num_heads=1, num_layers=2, ff_dim=128, cnf_hidden=128, cnf_steps=25):
    super().__init__()
    self.input_dim = input_dim
    self.embed_dim = embed_dim

    #FIXME
    # same embedding + transformer encoder as the autoregression model
    #self.embed = nn.Linear(input_dim, embed_dim)
    #encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=0.1, batch_first=True)
    #self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    # project hidden -> context vector for CNF
    #self.ctx = nn.Linear(embed_dim, embed_dim)

    #remove steps
    self.cnf = CNFDynamics(x_dim=input_dim, c_dim=0, hidden_dim=cnf_hidden, steps=cnf_steps)

  def forward(self, x):
    # Safe forward for debugging / torchinfo; does NOT compute likelihood.
    t = torch.zeros( x.shape[0], 1, device=x.device, dtype=x.dtype)
    return self.cnf.vf(x, t)

  def nll_loss(self, x):
    return -self.cnf(x)

  @torch.no_grad()
  def generate(self, out_dimensions):

    samples=self.cnf.generate(batch_size=out_dimensions[0])
    samples = samples.view(out_dimensions)
    return samples

#==============================================================================================
# Continuous flow matching
class FlowMatching(nn.Module):
    def __init__( self, x_dim, c_dim=0, hidden_dim=128, steps=50,):
        super().__init__()

        self.x_dim=x_dim
        self.steps=steps

        self.vf=VectorFieldNN( x_dim, c_dim, hidden_dim=hidden_dim)
        self.register_buffer( "time_grid", torch.linspace(0,1,steps+1))

    def loss(self, x1):
        B=x1.shape[0]
        device=x1.device

        x0=torch.randn_like(x1)

        t=torch.rand(B,1,device=device)

        # probability path

        xt=(1-t)*x0+t*x1

        #sigma=1e-4
        #eps=torch.randn_like(x1)
        #xt=(1-(1-sigma)*t)*x0+t*x1+sigma*eps

        # target velocity

        ut=x1-x0

        pred=self.vf(xt,t)

        loss=((pred-ut)**2).sum(-1).mean()

        return loss

    @torch.no_grad()
    def generate(self,batch_size):
        device=next(self.parameters()).device

        x=torch.randn( batch_size, self.x_dim, device=device)

        t=self.time_grid.to(device)

        dt=1/self.steps

        for k in range(len(t)-1):
            k1=self.vf(x,t[k])
            k2=self.vf( x+0.5*dt*k1, t[k]+0.5*dt)
            k3=self.vf( x+0.5*dt*k2, t[k]+0.5*dt)
            k4=self.vf( x+dt*k3, t[k]+dt)

            x=x+dt*(k1+2*k2+2*k3+k4)/6

        return x

class model_FM(nn.Module):
    def __init__( self, input_dim, hidden_dim=128, steps=50):
        super().__init__()

        self.fm=FlowMatching( x_dim=input_dim, hidden_dim=hidden_dim, steps=steps)

    def forward(self,x):
        t=torch.zeros( x.shape[0], 1, device=x.device, dtype=x.dtype)

        return self.fm.vf(x,t)

    def nll_loss(self,x):
        return self.fm.loss(x)

    @torch.no_grad()
    def generate(self,out_dimensions):

        samples=self.fm.generate( batch_size=out_dimensions[0])

        return samples.view(out_dimensions)

#==============================================================================================
# Custom vanillar NFlows, not working great
class realnvp_coupling_layer(nn.Module):
    def __init__(self, input_dim, latent_dim=256, mask=None):
        super().__init__()

        #If you give it a mask, treats in 1-flow, otherwise splits into 2 halves manually
        if mask==None:
            self.half_dim = input_dim // 2
        else:
            self.half_dim = input_dim

        #make the scale and translation networks
        self.scale_net = nn.Sequential(nn.Linear(self.half_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, self.half_dim))
        self.translate_net = nn.Sequential(nn.Linear(self.half_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, self.half_dim))

        self.register_buffer("mask", mask)

    def forward(self, x):

        #forward coupling: z_1=x_1   z_2=x_2*exp(s(x_1)) + t(x_1)
        #jacobean is J=exp(s(x_1))
        if self.mask==None:
            x1 = x[:, :self.half_dim]
            x2 = x[:, self.half_dim:]

            s = 2.0 * torch.tanh(self.scale_net(x1))
            t = self.translate_net(x1)

            z2 = x2 * torch.exp(torch.clamp(s,-5,5)) + t
            z = torch.cat([x1, z2], dim=1)
            log_det = s.sum(dim=1)
            return z, log_det, s

        else:
            x_mask = x * self.mask
            s = 2.0 * torch.tanh(self.scale_net(x_mask)) #- x_mask # Scale
            t = self.translate_net(x_mask) #- x_mask # Translation
            z = x_mask + (1 - self.mask) * (x * torch.exp(torch.clamp(s,-5,5)) + t)
            log_det = torch.sum(s * (1 - self.mask), dim=1)  # Log determinant

            return z, log_det, s

    def inverse(self, z):

        #Inverse is simple as well: x_1=z_1   x_2= (y_2-t(y_1))*exp(-s(y_1))
        if self.mask==None:
            z1 = z[:, :self.half_dim]
            z2 = z[:, self.half_dim:]

            s = 2.0 * torch.tanh(self.scale_net(z1))
            t = self.translate_net(z1)

            x2 = (z2 - t) * torch.exp(-s)
            x = torch.cat([z1, x2], dim=1)
            return x
        else:
            z_mask = z * self.mask
            s = 2.0 * torch.tanh(self.scale_net(z_mask)) #+ z_mask
            t = self.translate_net(z_mask) #+  z_mask
            x = z_mask + (1-self.mask)*(z - t)*torch.exp(-s)
            return x

class FlowBatchNorm(nn.Module):

    def __init__(self, feature_dim, eps=1e-5):
        super().__init__()

        #cap the sigma if needed
        self.eps = eps

        #learned parameters
        self.gamma = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, feature_dim))

    def forward(self, x):

        """
        x: (B,O,F)
        """
        B, O, F = x.shape

        mean = x.mean(dim=[0,1], keepdim=True)
        var = x.var(dim=[0,1], keepdim=True)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = torch.exp(self.gamma) * x_hat + self.beta

        log_det = O * self.gamma.sum()
        return y, log_det

    def inverse(self, y):

        mean = y.mean(dim=[0,1], keepdim=True)
        var = y.var(dim=[0,1], keepdim=True)

        x_hat = (y - self.beta) * torch.exp(-self.gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        return x

class model_normalizing_flow_custom(nn.Module):
    def __init__(self, input_dim, num_flows=10, latent_dim=256, doMask=True, doBatchNorm=False):
        super().__init__()

        self.input_dim = input_dim
        self.doMask = doMask
        self.doBatchNorm=doBatchNorm

        #Make alternating mask and add in flow layers
        if self.doMask:
            masks = [torch.ones(self.input_dim) for _ in range(num_flows)]
            for ii in range(num_flows):
                masks[ii][::2] = 0 #Every second
                #for jj in range(len(masks)//2): #every pair
                #    masks[ii][4*jj]=0
                #    masks[ii][4*jj+1]=0
                if ii%2==0: masks[ii]=1-masks[ii]
                print("Mask",ii,masks[ii], (masks[ii] == 0).sum())
            self.layers = nn.ModuleList([ realnvp_coupling_layer(input_dim,latent_dim,masks[ii]) for ii in range(num_flows) ])
        else:
            self.layers = nn.ModuleList([ realnvp_coupling_layer(input_dim,latent_dim) for ii in range(num_flows) ])

        self.norms = nn.ModuleList([ FlowBatchNorm(input_dim) for _ in range(num_flows) ])

    def forward(self, x):
        log_det_total = 0
        l2_norm=0
        z = x

        #loop over all the flow layers,  p(z)= p(f_1^{-1}(f_2^{-1}(...(x)))) |det J_1 + det J_2 +....|
        for layer,norm in zip(self.layers,self.norms):
            z, log_det, s = layer(z)
            if self.doBatchNorm: #if batchnorm make sure to also propage this transform
                z, log_det2 = norm(z)
                log_det_total+=log_det2
            if not self.doMask: #if alternating then flip dimension every layer
                z = torch.flip(z, dims=[1]) 
            log_det_total += log_det
            l2_norm+=s.pow(2) #Store a L2 regulator for the sclae

        return z, log_det_total, l2_norm

    def inverse(self, z):
        x = z

        #Go backwards through the layers
        for layer,norm in reversed(list(zip(self.layers,self.norms))):
            if not self.doMask:
                x = torch.flip(x, dims=[1]) # flip dimensions after each layer
            if self.doBatchNorm:
                x = norm.inverse(x)
            x = layer.inverse(x)

        return x

    def nll_loss(self, x):
        #if  p(z)= p(z)= p(f_1^{-1}(f_2^{-1}(...(x)))) |det J_1 + det J_2 +....|
        #then nll = \sum -0.5(z_i^2 + log(2pi)) - |det J_i|
        z, log_det, l2_norm = self.forward(x)
        log_pz = -0.5 * (z.pow(2) + torch.log(torch.tensor(2 * torch.pi)) ).sum(dim=1)
        return -log_pz - log_det + 1e-4*l2_norm.sum(dim=1)

    @torch.no_grad()
    def generate(self,out_dimensions):
        with torch.no_grad():
            z = torch.randn(out_dimensions[0],out_dimensions[1]*out_dimensions[2])
            samples = self.inverse(z)
            samples = samples.view(out_dimensions)
            return samples


#==============================================================================================
# Vanilla NFlows
class StableScaleNet(nn.Module): #just a clamped mlp
    def __init__(self, dim, hidden):
        super().__init__()

        self.net = nf.nets.MLP([dim, hidden, hidden, dim], init_zeros=True,)

    def forward(self, x):
        # Clamp scaling factors to 2sigma
        return 2.0 * torch.tanh(self.net(x))

class model_normalizing_flow(nn.Module): #Using normflows package
    #Normalizing flows, if have a invertible function x=f(z), and z~Gaus(z) then p(z)=p(f^{-1}(x)) |det J| (aka change of variables). Can compose multiple function together in a flow and the only thing to keep track is the composed final function f=f_1\cdotf_2... and the prod of the jacobian detetermints. Trick is then using neural networks to learn the functions, but make sure everything invertible
    def __init__(self, input_dim, num_flows=10, latent_dim=256, flow_type="neuralspline_coupling"):
        super().__init__()

        self.input_dim=input_dim

        # Base distribution
        base = nf.distributions.base.DiagGaussian(input_dim)

        flows = []
        for i in range(num_flows):
            #RealNVP: flips between update half the data with learned affine-connections (y=s(theta_1)*x+t(theta_2)) which are easy to invert. Learns the whole density p(x_1,x_2,...)
            if flow_type=="realnvp":
                # Alternating binary mask, swaps every layer
                mask = self.create_realnvp_mask(i)

                # scale/translation network
                #s_net = nf.nets.MLP([input_dim, latent_dim, latent_dim, input_dim], init_zeros=True)
                s_net = StableScaleNet(input_dim, latent_dim) #just a tanh clamp on output to 2sigma
                t_net = nf.nets.MLP([input_dim, latent_dim, latent_dim, input_dim], init_zeros=True)

                # RealNVP coupling layer
                #flows.append(nf.flows.ActNorm(self.input_dim))
                flows.append(nf.flows.MaskedAffineFlow(mask, t_net, s_net))
                #flows.append(nf.flows.ActNorm(self.input_dim))
            
            #Masked Autoregressive Flow: Applies a causal mask via MADE so that learns a product of conditionals on previous p(x_1,x_2,...)=prod p(x_i|p_{x<i}). Each of the transforms is still a affine, but input is all preceeding inputs (via mask) as opposed to half the input
            elif flow_type=="maf":
                num_blocks = 2
                #flows += [nf.flows.ActNorm(input_dim)]
                flows += [nf.flows.MaskedAffineAutoregressive(features=input_dim, hidden_features=latent_dim, num_blocks=2)]
                #flows += [nf.flows.LULinearPermute(input_dim)]
                #flows.append(nf.flows.MaskedAffineAutoregressive(features=input_dim, hidden_features=latent_dim, num_blocks=2))
                #flows.append(nf.flows.LULinearPermute(input_dim))

            #For both above, replaces the learned linear transform, with a ratio of two quadratic splites
            elif flow_type=="neuralspline_coupling":
                hidden_layers = 2 #depth of the MLP, latent_dim is its width
                num_bins = 8 #how many spline segments are used, bigger means more fine-tuned curves
                tail_bound = 3.0 #after this range makes linear transform, changes the impact of tails
                flows += [nf.flows.CoupledRationalQuadraticSpline(num_input_channels=input_dim, num_blocks=hidden_layers, num_hidden_channels=latent_dim, num_bins=num_bins, tail_bound=tail_bound)]
                flows += [nf.flows.LULinearPermute(input_dim)] #learned invertible linear transforms (y=Wx, with W learned) from glow, which gives better permitations of the coupling layers
            elif flow_type=="neuralspline_autoregressive":
                hidden_layers = 2
                num_bins = 8
                tail_bound = 3.0
                flows += [nf.flows.AutoregressiveRationalQuadraticSpline(num_input_channels=input_dim, num_blocks=hidden_layers, num_hidden_channels=latent_dim, num_bins=num_bins, tail_bound=tail_bound)]
                flows += [nf.flows.LULinearPermute(input_dim)]

        # Full normalizing flow model
        self.flow = nf.NormalizingFlow(base, flows)

    def create_realnvp_mask(self, layer_idx):
        mask = torch.zeros(self.input_dim)
        if layer_idx % 2 == 0: mask[::2] = 1.0
        else: mask[1::2] = 1.0
        return mask

    def forward(self, x):
        """
        Compute log probability of data.
        """
        return self.flow.log_prob(x)

    def nll_loss(self, x):
        """
        Compute log likelihood.
        """
        return -self.flow.log_prob(x)

    def generate(self, out_dimensions):
        """
        Generate samples.
        """
        samples, log_prob = self.flow.sample(out_dimensions[0])
        return samples.view(out_dimensions)

################################
# Diffusion model
class DiffusionMLP(nn.Module):
    '''
    Simple MLP noise predictor for diffusion. Learning cumulative added noise epsilon(x_{t-1}, t)
    '''
    def __init__(self, input_dim, hidden_dim=256, time_embed_dim=128):
        super().__init__()

        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),)

        #self.mlp = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, input_dim),) 
        self.mlp = nn.Sequential(nn.Linear(input_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, input_dim),)

    def sinusoidal_embedding(self, t, dim):
        """
        Standard sinusoidal timestep embedding. Converts t -> [sin(w_0 t),cos(w_0 t),sin(w_1 t),cos(w_1 t), ...] (check what is w_i)
        Can also do a normal embedding?
        t: (B,)
        """
        device = t.device
        half_dim = dim // 2

        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)

        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb
     
    def forward(self, x, t):
        """
        x: (B, D)
        t: (B,) integer timesteps
        """

        #Add the sinusioidal time conditioner
        #t_emb = self.sinusoidal_embedding(t.float(), self.time_embed_dim)
        #t_emb = self.time_mlp(t_emb)

        #Linear timestep instead
        ##t_emb = t.float().view(-1, 1)
        t_emb = t.float().unsqueeze(-1)
        t_emb = t_emb / 1000.0 # normalize by max timestep

        h = torch.cat([x, t_emb], dim=-1)

        return self.mlp(h)

# ------------------------------------------------------------
# Diffusion model
# ------------------------------------------------------------
class model_diffusion(nn.Module):
    """
    Basic DDPM-style diffusion model

    Consider diffusion/forward processa s function compostion   q(x_{1:T}|x_0)=prod^T q(x_t|x_{t-1})                where q(x_t|x_{t-1})=Gaus(sqrt(1-beta_t),beta_t))
    and allows a closed form at step t                          q(x_t|x_0)=Gaus(sqrt(baralpha_t)*x0,1-baralpha_t)    where alpha_t=1-beta_t and baralpha_t=prod^t alpha_t
    We can also re-paramtrize this as                           q(x_t|x_0)=sqrt(baralpha_t)*x0+(1-baralpha_t)*epsilon for epsilon=Gaus(0,1)
    Note need increasing noisce beta_1 < beta_2 < ...

    The reverse process is only defined when condtioned on x_0  q(x_{t-1}|x_t,x_0)=Gaus(tildemu_t(x_t,x_0),tildebeta_t)    with complicated expresions for tildemu and tildebeta
    Want to learn reverse process                               p_theta(x_{0:T})=p(x_0) prod^T p_theta(x_{t_1}|x_t)      where p(x_{t-1}|x_t)=Gaus(mu_theta(x_t,t),beta_t^2)
    Note setting the covariance matrix to beta_t^2 is a choice
    Throught some tricky math, solution is to maximize D_KL(q(x_{t-1}|x_t,x_0)||p(x_{t-1}|x_t)) which is a simplification of E|-log(p_theta(x_{0:T})/q(x_{1:T}|x_0))|
    With a bunch of simplifications can show the loss is        L=E_x0,t,epsilon|epsilon-epsilon_theta((baralpha_t)*x0+(1-baralpha_t)*epsilon, t )|^2

    Update rules is                             x_{t-1}=1/sqrtalpha_t(x_t - (1-alpha_t)/(sqrt(1-baralpha_t))*epsilon_theta) + sqrt(beta)*z    where z=Gaus(0,1)

    In DDIM update rules is determinitistic     x_{t_1}= sqrt(alpha_{t-1}) hat x_0 + sqrt(1-alpha_{t-1})*epsilon_theta   where x_0=sqrt(alpha_{t-1}/alpha_t)*(x_t-sqrt(1-alpha_t)*epsilon_theta)
        
    """
    def __init__(self, input_dim, hidden_dim=256, timesteps=1000, beta_start=1e-4, beta_end=2e-2,mode="DDPM"):
        super().__init__()

        self.input_dim = input_dim
        self.timesteps = timesteps
        self.mode      = mode

        # Noise predictor network
        self.epsilon_model = DiffusionMLP(input_dim=input_dim, hidden_dim=hidden_dim,)

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Register buffers so they move with model.to(device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    # --------------------------------------------------------
    # Forward diffusion process q(x_t | x_0)
    # --------------------------------------------------------
    def q_sample(self, x0, t):
        """
        Add noise to clean samples.
        """

        #Random gaussian noise
        noise = torch.randn_like(x0)

        #Cumulated values to this time-step
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t].unsqueeze(-1))
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1-self.alpha_bar[t].unsqueeze(-1))

        #q(x_t|x_0)=sqrt(baralpha_t)*x0+(1-baralpha_t)*epsilon for epsilon=Gaus(0,1)
        xt = (sqrt_alpha_bar_t * x0 +sqrt_one_minus_alpha_bar_t * noise)

        return xt, noise

    # --------------------------------------------------------
    # Training objective
    # --------------------------------------------------------
    def forward(self, x):
        """
        Compute diffusion training loss.
        """

        batch_size = x.shape[0]
        device = x.device

        # Random timestep per sample
        t = torch.randint(0, self.timesteps, (batch_size,), device=device,)

        # Add noise to the sample
        xt, noise = self.q_sample(x, t)

        # Predict cumultant noise
        noise_pred = self.epsilon_model(xt, t)

        # Standard DDPM objective |\epsilon-\epsilon_theta|
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def mse_loss(self, x):
        """
        Kept for API compatibility with your flow model.
        """
        return self.forward(x)

    # --------------------------------------------------------
    # Reverse diffusion sampling
    # --------------------------------------------------------
    @torch.no_grad()
    def generate(self, out_dimensions):
        """
        Generate samples via reverse diffusion.
        """

        batch_size = out_dimensions[0]
        device = self.betas.device

        #Sample from final gaussian space x_T~N(0,I) 
        x = torch.randn(batch_size, self.input_dim, device=device,)

        #Should speed up DDPM which can use coarser steps then in the forward process

        #Loop over timesteps, and update
        for t in reversed(range(self.timesteps)):

            #Get the injected noise beta/alpha
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long,)
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]

            # Predict noise
            epsilon_theta = self.epsilon_model(x, t_batch)

            if self.mode=="DDPM":
                # DDPM reverse step
                # x_{t-1}=1/\sqrt\alpha_t(x_t - (1-\alpha_t)/(sqrt(1-\bar\alpha_t))*\epsilon_theta) + \sqrt(\beta)*z where z=Gaus(0,1)
                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t))
                mean = coef1 * (x - coef2 * epsilon_theta)

                #if t>0 add the random noise to sample
                if t > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(beta_t)
                    x = mean + sigma * noise
                #else just start the process
                else:
                    x = mean

            elif self.mode=="DDIM":
                # DDIM reverse step
                # x_{t_1}= sqrt(alpha_{t-1}) hatx_0 + sqrt(1-alpha_{t-1})*epsilon_theta   where hatx_0=(x_t-sqrt(1-alpha_t)*epsilon_theta)/sqrt(alpha_t)
                x0_pred = (x - torch.sqrt(1 - alpha_t) * eps_theta) / torch.sqrt(alpha_t)

                alpha_tm1 = self.alpha[t-1]
                x = (torch.sqrt(alpha_tm1) * x0_pred + torch.sqrt(1 - alpha_tm1) * epsilon_theta)

        return x.view(out_dimensions)

###############################
# Score based Stochastic differential equation
###############################

class ScoreNet(nn.Module): #Same as DiffusionMLP above!
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(input_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, input_dim),)

    def forward(self, x, t):

        t = t.unsqueeze(-1)
        h = torch.cat([x, t], dim=-1)

        return self.net(h)

class model_score_SDE(nn.Module):
    '''
    Learn the score-function via stochastic differential equation (variance preserving form)
    Whole model is              dx=f(x,t)dt+g(t)dw      for Weiner process w
    We denote  p_t(x) is the density of x(t)
    Reverse process is then     dx=(f(x,t)-g(t)^2 nabla_x log(p_t(x)) )dt+g(t)dw   for reverse time and reverse dw

    Train network to learn score via loss L= E_x,t( lambda(t)*|s(x(t),t) - nabla_x log(p_{0t}(x(t)|x(0)))|^2 ) for transition probability p_{0t}(x(t)|x(0))=Gauss which is very good approximation of p_t(x)

    The variance-preserving (DDPM not score-based) appoarach models: x_t=sqrt(1-beta_t)*x_{t-1}+sqrt(beta_t)*epsilon
    hence                       dx=0.5beta(t)x*dt+sqrt(beta(t))*dw
    
    The cumulative solutions follows:           x_t=alpha(t)*x_0+sigma(t)*epsilon
    Where:  alpha(t)=exp(-0.5 int_0^t beta(s) ds    (by solving ODE ignoreing noise)
            sigma(t)=sqrt(1-alpha(t))               (same as in discrete version where alpha(t) is like sqrt(bar alpha))
    and so p_{0t}(x(t)|x(0))=Gaus(x_t|alpha_t*x0,sigma_t) and nabla_x log(p_{0t}) = -epsilon/sigma(t)
    DDPM use lambda(t)=sigma_t^2 in loss which results in loss of L=|epsilon_theta-epsilon| since training gives s(x,t)=-epsilon_theta(x,t)/sigma(t)
    '''

    def __init__(self, input_dim, hidden_dim=256, beta_min=0.1, beta_max=20.0, mode="sde"):
        super().__init__()

        #Will use linear beta scheduler
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.score_model = ScoreNet(input_dim=input_dim, hidden_dim=hidden_dim,)

        self.mode=mode
        self.input_dim = input_dim

    def marginal_prob(self, x0, t):

        #Use linear schedule: beta(t)=beta_min+t*(beta_max - beta_min)
        #alpha alpha(t)=exp(-0.5 int_0^t beta(s) ds
        alpha = torch.exp(-0.25 * (self.beta_max - self.beta_min) * t**2 - 0.5 * self.beta_min * t)

        #x_t=alpha(t)*x_0+sigma(t)*epsilon
        mean    = alpha.unsqueeze(-1) * x0
        sigma   = torch.sqrt(1.0 - alpha**2)
        epsilon = torch.randn_like(x0)

        #Return each component
        return mean, sigma, epsilon

    def forward(self, x0):

        B = x0.shape[0]
        device = x0.device

        #Avoid t=0 in case sigma->0
        eps=1e-5
        t = eps + (1 - eps) * torch.rand(B, device=device)

        #Derive x_t=alpha(t)*x_0+sigma(t)*epsilon
        mean, sigma, epsilon = self.marginal_prob(x0, t)
        xt = mean + sigma.unsqueeze(-1) * epsilon

        #Get s(x,t)
        score = self.score_model(xt, t)

        '''
        #Loss is sigma**2 |s(x,t)-eps/sigma|
        target = -epsilon / sigma.unsqueeze(-1)
        loss = (sigma[:, None]**2 * (score - target)**2).mean()
        '''

        #Loss is |epsilon_theta-eps|
        target = -epsilon
        pred = sigma[:, None] * score
        loss = F.mse_loss(pred, target)

        return loss

    def mse_loss(self, x0):
        return self.forward(x0)

    @torch.no_grad()
    def generate(self, out_dimensions, num_steps=1000,):
        '''
        Run Euler–Maruyama forward numerical integration
        dx=(-0.5beta(t)-beta(t) nabla_x log(p_t(x)) )dt + sqrt(beta(t)) dw
        '''

        batch_size = out_dimensions[0]

        device = next(self.parameters()).device
        x = torch.randn(batch_size, self.input_dim, device=device,)

        #Loop over timesteps
        dt = -1.0 / num_steps
        for i in range(num_steps):

            #backwards time
            t = torch.ones(batch_size, device=device) * (1 - i / num_steps)
            beta_t = (self.beta_min + t * (self.beta_max - self.beta_min))

            #Score = nabla_x log(p_t(x))  
            score = self.score_model(x, t)

            #If solving via stoachatsic equation
            if self.mode=="sde":
                #Drift-term = (-0.5beta(t)-beta(t) nabla_x log(p_t(x)) )
                drift = (-0.5 * beta_t.unsqueeze(-1) * x - beta_t.unsqueeze(-1) * score)

                #diffusion term=sqrt(beta(t))
                diffusion = torch.sqrt(beta_t)

                #numericallu intergate via Euler–Maruyama 
                z = torch.randn_like(x)
                x = (x + drift * dt + diffusion.unsqueeze(-1) * torch.sqrt(torch.abs(torch.tensor(dt))) * z)

            #or solving determinisitically
            elif self.mode=="ode":
                #Drift-term = (-0.5beta(t)-0.5*beta(t) nabla_x log(p_t(x)) )
                drift = (-0.5 * beta_t.unsqueeze(-1) * x - 0.5 * beta_t.unsqueeze(-1) * score)

                #numerically intergate via Euler
                x = (x + drift * dt )

        return x.view(out_dimensions)

###############################
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

def _torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

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
    model = _torch_load(model_path, map_location="cpu")
    return model

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

def build_unbinned_model(input_dim, args_or_dict):
    get = lambda key, default=None: _config_get(args_or_dict, key, default)
    if _as_bool(get("cnf", False)):
        return model_CNF(
            input_dim=input_dim[1]*input_dim[2],
            embed_dim=_as_int(get("embed_dim", 256), 256),
            num_heads=_as_int(get("num_heads", 1), 1),
            num_layers=_as_int(get("num_layers", 2), 2),
            ff_dim=_as_int(get("ff_dim", 128), 128),
            cnf_hidden=_as_int(get("cnf_hidden", get("flow_hidden", 128)), 128),
            #cnf_steps=_as_int(get("cnf_steps", 8), 8),
        )
    if _as_bool(get("fm", False)):
        return model_FM(
            input_dim=input_dim[1]*input_dim[2],
            hidden_dim=_as_int(get("embed_dim", 256), 256),
            #num_heads=_as_int(get("num_heads", 1), 1),
            #num_layers=_as_int(get("num_layers", 2), 2),
            #ff_dim=_as_int(get("ff_dim", 128), 128),
            steps=_as_int(get("cnf_steps", 50), 50),
        )
    if _as_bool(get("mdn", False)):
        return model_autoregressive_transformer_MDN(
            input_dim=input_dim[2],
            n_mix=_as_int(get("n_mix", 25), 25),
            embed_dim=_as_int(get("embed_dim", 256), 256),
            num_heads=_as_int(get("num_heads", 1), 1),
            num_layers=_as_int(get("num_layers", 2), 2),
            ff_dim=_as_int(get("ff_dim", 128), 128),
        )
    if _as_bool(get("nf", False)):
        return model_normalizing_flow(
            input_dim=input_dim[1]*input_dim[2],
            num_flows=6,
            latent_dim=128,
        )
    if _as_bool(get("diff", False)):
        return model_diffusion(input_dim=input_dim[1]*input_dim[2], hidden_dim=256)
    if _as_bool(get("sde", False)):
        return model_score_SDE(input_dim=input_dim[1]*input_dim[2], hidden_dim=256)
    return model_autoregressive_transformer(
        input_dim=input_dim[2],
        embed_dim=_as_int(get("embed_dim", 256), 256),
        num_heads=_as_int(get("num_heads", 1), 1),
        num_layers=_as_int(get("num_layers", 2), 2),
        ff_dim=_as_int(get("ff_dim", 128), 128),
    )

def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    args,
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
    saved_what = "checkpoint" if save_mode == "checkpoint" else "model state"
    print(f"Saved {saved_what} to {ckpt_path}", flush=True)
    return ckpt_path

def load_unbinned_model_for_plot(model_path, input_dim, device="cpu"):
    obj = _torch_load(model_path, map_location="cpu")
    metadata = {}

    if isinstance(obj, dict) and "model_state_dict" in obj:
        ckpt_args = obj.get("args", {})
        metadata.update(ckpt_args)
        for key in (
            "artifact_type",
            "save_mode",
            "epoch",
            "epoch_display",
            "loss",
            "best_epoch",
            "best_epoch_display",
            "best_loss",
            "model_mode",
            "scheduler",
            "optimizer",
            "weight_decay",
            "grad_clip",
            "current_lr",
            "next_lr",
        ):
            if key in obj:
                metadata[key] = obj[key]
        metadata["resolved_model_mode"] = obj.get("model_mode", resolved_model_mode(ckpt_args))
        model = build_unbinned_model(input_dim, ckpt_args)
        model.load_state_dict(obj["model_state_dict"])
    else:
        model = obj

    model.to(device)
    model.eval()
    return model, metadata

def model_has_nonfinite_parameters(model):
    for tensor in model.state_dict().values():
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
            return True
    return False

def infer_log_dir_from_path(model_path):
    parent = os.path.dirname(os.path.abspath(model_path))
    if os.path.basename(parent) == "checkpoints":
        return os.path.dirname(parent)
    return parent

def parse_arguments_txt(txt_path):
    meta = {}
    if not os.path.exists(txt_path):
        return meta
    with open(txt_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
                continue
            parts = line.split()
            if len(parts) >= 2:
                meta[parts[0].strip()] = " ".join(parts[1:]).strip()
    return meta

def _format_scalar(value):
    as_float = _as_float(value, None)
    if as_float is None:
        return str(value)
    return f"{as_float:g}"

def _display_epoch_from_meta(meta, key):
    display = _as_int(meta.get(f"{key}_display", None), None)
    if display is not None:
        return display
    index = _as_int(meta.get(key, None), None)
    if index is None:
        return None
    return index + 1

def _artifact_label_from_name(name):
    if name is None:
        return None
    base = os.path.splitext(os.path.basename(str(name)))[0]
    if base == "best":
        return "best"
    match = re.search(r"epoch[_-](\d+)", base)
    if match is not None:
        return f"ep {int(match.group(1)) + 1}"
    return base if base else None

def build_run_caption(meta, fallback=None):
    model_mode = meta.get("resolved_model_mode", meta.get("model_mode", "model"))
    batch_size = meta.get("batch_size", meta.get("batch-size", "?"))
    lr = meta.get("lr", "?")
    epochs = meta.get("epochs", meta.get("total_epochs", None))
    scheduler = meta.get("scheduler", "none")
    input_format = meta.get("input_format", meta.get("input-format", None))
    n_mix = meta.get("n_mix", meta.get("n-mix", None))
    best_epoch_display = _display_epoch_from_meta(meta, "best_epoch")
    epoch_display = _display_epoch_from_meta(meta, "epoch")

    artifact_label = None
    fallback_label = _artifact_label_from_name(fallback)
    if fallback_label == "best":
        artifact_label = f"best ep {best_epoch_display}" if best_epoch_display is not None else "best"
    elif epoch_display is not None:
        artifact_label = f"ep {epoch_display}"
    else:
        artifact_label = fallback_label

    first_line = [str(model_mode)]
    if artifact_label is not None:
        first_line.append(artifact_label)
    if best_epoch_display is not None and (artifact_label is None or not artifact_label.startswith("best")):
        first_line.append(f"best ep {best_epoch_display}")

    second_line = []
    if input_format is not None:
        second_line.append(str(input_format))
    if n_mix is not None and _as_bool(meta.get("mdn", False)):
        second_line.append(f"{n_mix} Gauss")
    second_line.extend([f"bs {batch_size}", f"lr {_format_scalar(lr)}"])
    if epochs is not None:
        second_line.append(f"tot {epochs}")

    lines = [" ".join(first_line), " ".join(second_line)]
    if scheduler not in (None, "none"):
        sched_label = {
            "cos_damping": "cos damp",
            "cosine": "cos",
            "plateau": "plateau",
        }.get(str(scheduler), str(scheduler))
        sched_line = [f"sched {sched_label}"]
        if scheduler == "cos_damping":
            start_epoch = meta.get("cos_damping_start_epoch", meta.get("cos_start_epoch", None))
            start_display = _epoch_display_from_index(start_epoch)
            final_lr = meta.get("cos_damping_final_lr", meta.get("cos_final_lr", None))
            if start_display is not None:
                sched_line.append(f"start ep {start_display}")
            if final_lr is not None:
                sched_line.append(f"lr_f {_format_scalar(final_lr)}")
        elif scheduler in ("cosine", "plateau") and meta.get("scheduler_min_lr", None) is not None:
            sched_line.append(f"min lr {_format_scalar(meta.get('scheduler_min_lr'))}")
        lines.append(" ".join(sched_line))
    return "\n".join(lines)

def _parse_caption_fields(label):
    text = str(label).replace("\n", " ")
    fields = {}
    parts = str(label).splitlines()
    if len(parts) > 0:
        first_tokens = parts[0].split()
        if len(first_tokens) > 0:
            fields["model"] = first_tokens[0]

    # Captions for intermediate checkpoints can include both the plotted
    # checkpoint and the run's best epoch, e.g. "model ep 1 best ep 7".
    # The comparison title should describe the plotted stage first.
    match = re.search(r"(?<!best )\bep\s+\d+", text)
    if match:
        fields["checkpoint"] = match.group(0)
    else:
        match = re.search(r"\bbest ep\s+\d+", text)
        if match:
            fields["checkpoint"] = match.group(0)

    for key, pattern, label_text in (
        ("bs", r"\bbs\s+([^\s]+)", "bs"),
        ("lr", r"\blr\s+([^\s]+)", "lr"),
        ("tot", r"\btot\s+([^\s]+)", "tot"),
        ("min_lr", r"\bmin lr\s+([^\s]+)", "min lr"),
        ("lr_f", r"\blr_f\s+([^\s]+)", "lr_f"),
        ("start_ep", r"\bstart ep\s+([^\s]+)", "start ep"),
    ):
        match = re.search(pattern, text)
        if match:
            fields[key] = f"{label_text} {match.group(1)}"

    sched_match = re.search(r"\bsched\s+(.+?)(?:\s+start ep|\s+lr_f|\s+min lr|$)", text)
    if sched_match:
        fields["sched"] = f"sched {sched_match.group(1).strip()}"
    elif "cos damping" in text:
        fields["sched"] = "sched cos damping"
    elif "cosine" in text:
        fields["sched"] = "sched cosine"
    elif "plateau" in text:
        fields["sched"] = "sched plateau"

    return fields

def _caption_comparison(labels, first_run_idx=1):
    run_labels = list(labels[first_run_idx:])
    field_order = ["model", "checkpoint", "bs", "lr", "tot", "sched", "start_ep", "lr_f", "min_lr"]
    parsed = [_parse_caption_fields(label) for label in run_labels]

    common = []
    diff_labels = []
    for idx, fields in enumerate(parsed):
        items = []
        for key in field_order:
            values = [item.get(key) for item in parsed]
            present_values = [value for value in values if value not in (None, "")]
            if len(present_values) == len(parsed) and len(set(present_values)) == 1:
                continue
            if fields.get(key):
                items.append(fields[key])
        diff_labels.append("\n".join(items) if items else f"run {idx + 1}")

    if parsed:
        for key in field_order:
            values = [item.get(key) for item in parsed]
            present_values = [value for value in values if value not in (None, "")]
            if len(present_values) == len(parsed) and len(set(present_values)) == 1:
                common.append(present_values[0])

    return diff_labels, common

def _plot_note_from_common(common_items, unavailable_notes=None):
    note_lines = []
    if common_items:
        note_lines.append("Plot: " + "; ".join(common_items))
    if unavailable_notes:
        note_lines.extend(
            f"{str(label).replace(chr(10), ' ')}: unavailable ({reason})"
            for label, reason in unavailable_notes
        )
    return note_lines

def _wrapped_note(note_lines, width=150):
    if not note_lines:
        return ""
    return "\n".join(textwrap.wrap(" | ".join(note_lines), width=width, break_long_words=False))

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
    p.add_argument("--num-features", dest="num_features", type=int, default=2,
                   help="Feature dimension per constituent (default: 2 = [deltaR, kt])")
    p.add_argument("--num-bins", dest="num_bins", type=int, nargs="+", default=[50, 50],
                   help="Binning spec for compatibility with binned training scripts")

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
    p.add_argument("--multi-loss-plot", action="store_true", default=False,
                   help="Log multiple loss definitions without affecting main training")

    # misc
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="Force device; default auto")

    # logging / checkpointing
    p.add_argument("--log-dir", dest="log_dir", type=str, default="models/test",help="Logging directory")
    p.add_argument(
        "--plot-dir",
        "--plot-out-dir",
        dest="plot_dir",
        type=str,
        default=None,
        help="Output directory for plots. Default: use --log-dir / inferred checkpoint log directory",
    )
    p.add_argument("--save-mode", type=str, default="checkpoint", choices=["checkpoint", "model", "none"],
                   help="Saved training artifact: full checkpoint, model-state only, or none")
    p.add_argument("--contin", action="store_true", default=False,help="Continue training from a saved model")
    p.add_argument("--model-path", "--checkpoint", dest="model_path", type=str, nargs="+",default=[],help="Path(s) to model/checkpoint to load")
    
    # plotting options
    p.add_argument(
        "--hist2d-xrange",
        type=float,
        nargs=2,
        default=None,
        help="2D Lund histogram x range: xmin xmax",
    )
    p.add_argument(
        "--hist2d-yrange",
        type=float,
        nargs=2,
        default=None,
        help="2D Lund histogram y range: ymin ymax",
    )
    p.add_argument(
        "--hist2d-bins",
        type=int,
        nargs=2,
        default=[20, 20],
        help="2D Lund histogram bins: xbins ybins",
    )
    p.add_argument(
        "--hist2d-shape",
        "--hist2d_shape",
        "--hist2d-layout",
        dest="hist2d_shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("ROWS", "COLS"),
        help="Manual 2D Lund subplot shape. Default auto: 2 -> 1x2, 3 -> 1x3, 4 -> 2x2",
    )
    p.add_argument(
        "--plot-max-batches",
        type=int,
        default=None,
        help="Only plot this many validation batches. Default: plot all batches",
    )
    p.add_argument(
        "--hist1d-ranges",
        type=float,
        nargs="+",
        default=None,
        help="Flattened 1D ranges: kt_min kt_max dr_min dr_max",
    )
    p.add_argument(
        "--hist1d-bins",
        type=int,
        default=30,
        help="Number of bins for 1D histograms",
    )
    p.add_argument(
        "--hist1d-logy",
        action="store_true",
        default=False,
        help="Also save log-y 1D histograms",
    )
    p.add_argument(
        "--hist-ratio-diff",
        "--hist-diff-ratio",
        action="store_true",
        default=False,
        help="Also save generated-vs-original relative difference plots for multi-sample 1D/2D histograms",
    )
    p.add_argument(
        "--hist-ratio-min-count",
        type=int,
        default=5,
        help="Mask 2D relative-difference bins with fewer original entries than this",
    )
    p.add_argument(
        "--hist-ratio-vmax",
        type=float,
        default=1.0,
        help="Symmetric color limit for 2D fractional relative-difference plots",
    )
    return p.parse_args()

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

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_lundplane(input_vec, pad_length=15):

  #Assume input is dimensions [Nevents, Nconstituents, 4-vecs]
  jetDef10 = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0, fastjet.E_scheme)
  #jetDefCA = fastjet.JetDefinition1Param(fastjet.cambridge_algorithm, 10.0)

  lund_plane=[]
  for ii in range(input_vec.shape[0]):

    # Convert the constituent information into a format usable for fastjet (PseudoJet objects)
    constituents = []
    for jj in range(input_vec.shape[1]):
      constituents.append(fastjet.PseudoJet(float(input_vec[ii,jj,1]), float(input_vec[ii,jj,2]), float(input_vec[ii,jj,3]),float(input_vec[ii,jj,0])))

    # Run the jet clustering on the jet constituents using the anti-kt algorithm
    cs_akt = fastjet.ClusterSequence(constituents, jetDef10)
    inclusiveJets10 = fastjet.sorted_by_pt(cs_akt.inclusive_jets(25.))

    # Skip if inclusiveJets10 is empty
    if not inclusiveJets10: continue

    # Get Lund plane declusterings
    lundPlane = ljpHelpers.jet_declusterings(inclusiveJets10[0])
    lp_points=[]
    for kk in range(len(lundPlane)):
      if (lundPlane[kk].delta_R > 0 and lundPlane[kk].z > 0):
        dr_val = math.log(1.0 / lundPlane[kk].delta_R)
        kt_val = math.log(lundPlane[kk].kt)
        lp_points.append([kt_val,dr_val])

    # Free C++ memory
    constituents.clear()
    del cs_akt
    del inclusiveJets10
    del lundPlane

    #push back and clean-up
    while len(lp_points)<pad_length:
      lp_points.append([-1,-1])
    lund_plane.append(lp_points[:pad_length].copy())
    lp_points.clear()

  #pad the length
  lund_plane=np.asarray(lund_plane)
  return lund_plane


def projection_plot(inputs,labels=["original","generated","predicted"],outdir="./Plots/", unavailable_notes=None):

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
      axs[ii].hist(inputs[jj][:,ii],bins=20,range=[mins[ii],maxs[ii]],histtype="step",density=False,linestyle=linestyles[jj],label=labels[jj])
    if ii==0: axs[0].legend()
    axs[ii].set_yscale("log")

  if unavailable_notes:
    note_text = "Unavailable:\n" + "\n".join(
      f"{label}: {reason}" for label, reason in unavailable_notes
    )
    fig.text(0.98, 0.5, note_text, ha="right", va="center", fontsize=8)

  name="projection"
  fig.savefig(os.path.join(outdir,name+".png"))
  fig.savefig(os.path.join(outdir,name+".pdf"))
  plt.close(fig)

def lund_plot(
    inputs,
    labels=["original","generated","predicted"],
    outdir="./Plots/",
    hist2d_xrange=None,
    hist2d_yrange=None,
    hist2d_bins=(20, 20),
    hist2d_shape=None,
    hist2d_layout=None,
    unavailable_notes=None,
):

  #Get ranges
  Ndim=inputs[0].shape[1]
  Nin=len(inputs)
  unavailable_notes = unavailable_notes or []
  Nplots = Nin + len(unavailable_notes)
  use_compared_titles = Nplots > 1
  diff_labels, common_items = _caption_comparison(labels, first_run_idx=1) if use_compared_titles else ([], [])

  if not os.path.exists(outdir):
    os.makedirs(outdir)
  mins=np.zeros(Ndim)
  maxs=np.zeros(Ndim)
  for ii in range(Ndim):
    for jj in range(Nin):
      mins[ii]=min(mins[ii],np.min(inputs[jj][:,ii]))
      maxs[ii]=max(maxs[ii],np.max(inputs[jj][:,ii]))

  #FIXME 1=DR 0=kt
  mins[0]=-5
  maxs[0]=7
  mins[1]=-1
  maxs[1]=10

  #Make plot
  if Ndim>=2:
    # Create subplots for each input (original, generated, etc.)
    if hist2d_shape is None:
      hist2d_shape = hist2d_layout
    nrows, ncols = resolve_hist2d_shape(Nplots, hist2d_shape)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols + 1.0, 4.4 * nrows + 0.6),
        squeeze=False,
    )
    flat_axs = axs.ravel()
    used_axs = flat_axs[:Nplots]

    last_hist = None
    for jj in range(Nin):
        ax = used_axs[jj]
        # Determine plotting range
        x_range = hist2d_xrange if hist2d_xrange is not None else [mins[1], maxs[1]]
        y_range = hist2d_yrange if hist2d_yrange is not None else [mins[0], maxs[0]]

        # Plot 2D histogram
        last_hist = ax.hist2d(
            inputs[jj][:, 1],
            inputs[jj][:, 0],
            range=[x_range, y_range],
            bins=hist2d_bins,
            cmap="Blues",
            norm="log"
        )

        # --------------------------
        # Add titles and labels
        # --------------------------

        if use_compared_titles:
            if jj == 0:
                panel_label = "original"
            elif jj - 1 < len(diff_labels):
                panel_label = diff_labels[jj - 1]
            else:
                panel_label = f"run {jj}"
        else:
            panel_label = str(labels[jj]) if jj < len(labels) else f"sample_{jj}"
        ax.set_title(panel_label, pad=8, fontsize=10)

        # Axis labels
        ax.set_xlabel(r"$\log(1/\Delta R)$")
        ax.set_ylabel(r"$\log(k_t)$")

    for offset, (label, reason) in enumerate(unavailable_notes):
        ax = used_axs[Nin + offset]
        ax.set_title(label)
        ax.text(
            0.5,
            0.5,
            f"Plot unavailable\n{reason}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            wrap=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in flat_axs[Nplots:]:
      ax.set_visible(False)

    # --------------------------
    # Add colorbar (shared)
    # --------------------------
    if last_hist is not None:
      cbar = fig.colorbar(last_hist[3], ax=used_axs.tolist(), fraction=0.022, pad=0.055)
      cbar.set_label("Density (log scale)")

    fig.suptitle("Lund Plane Distribution", fontsize=14, y=0.985)
    caption_lines = _plot_note_from_common(common_items, unavailable_notes)
    bottom_margin = 0.13 if caption_lines else 0.09
    fig.subplots_adjust(left=0.08, right=0.88, bottom=bottom_margin, top=0.90, wspace=0.30, hspace=0.55)
    if caption_lines:
      fig.text(
        0.08,
        0.025,
        _wrapped_note(caption_lines, width=150),
        ha="left",
        va="bottom",
        fontsize=8,
      )

    # Save
    name = "lund"
    fig.savefig(os.path.join(outdir, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)

def resolve_hist2d_shape(nplots, hist2d_shape=None):
  if hist2d_shape is not None:
    nrows, ncols = hist2d_shape
    if nrows < 1 or ncols < 1:
      raise ValueError("--hist2d-shape values must be positive")
    if nrows * ncols < nplots:
      raise ValueError(
          f"--hist2d-shape {nrows} {ncols} has only {nrows * ncols} slots for {nplots} plots"
      )
    return nrows, ncols

  if nplots <= 3:
    return 1, nplots
  if nplots == 4:
    return 2, 2

  ncols = math.ceil(math.sqrt(nplots))
  nrows = math.ceil(nplots / ncols)
  return nrows, ncols

def resolve_hist2d_layout(nplots, hist2d_layout=None):
  return resolve_hist2d_shape(nplots, hist2d_layout)

def _default_hist1d_ranges(inputs, display_order, quantiles=(0.5, 99.5), margin_fraction=0.05):
    ranges = []
    reference = inputs[0]
    for ii in range(reference.shape[1]):
        feature_idx = display_order[ii] if ii < len(display_order) else ii
        values = reference[:, feature_idx]
        values = values[np.isfinite(values)]
        if values.size == 0:
            ranges.append([-1.0, 1.0])
            continue

        vmin, vmax = np.percentile(values, quantiles)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = np.min(values)
            vmax = np.max(values)

        margin = margin_fraction * (vmax - vmin)
        if margin == 0:
            margin = 1.0
        ranges.append([vmin - margin, vmax + margin])
    return ranges

def _symmetric_limit(values, fallback=1.0):
    finite = []
    for value in values:
        arr = np.asarray(value)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            finite.append(np.abs(arr))
    if not finite:
        return fallback
    merged = np.concatenate(finite)
    if merged.size == 0:
        return fallback
    vmax = np.percentile(merged, 99.0)
    vmax = max(vmax, np.max(merged) if vmax == 0 else vmax)
    return vmax if vmax > 0 else fallback

def plot_combined_1dhist_ratio_diff(
    inputs,
    labels=None,
    out_dir="./Plots/",
    hist1d_ranges=None,
    hist1d_bins=30,
    logy=False,
    out_name=None,
    unavailable_notes=None,
    min_reference_count=5,
    max_abs_diff=1.0,
):
    if len(inputs) <= 1:
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if labels is None:
        labels = [f"sample_{i}" for i in range(len(inputs))]
    diff_labels, common_items = _caption_comparison(labels, first_run_idx=1)
    comparison_labels = diff_labels if len(diff_labels) > 0 else labels[1:]

    Ndim = inputs[0].shape[1]
    display_order = [0, 1] if Ndim >= 2 else list(range(Ndim))

    if hist1d_ranges is None:
        hist1d_ranges = _default_hist1d_ranges(inputs, display_order)

    fig, axs = plt.subplots(Ndim, 1, figsize=(8.0, 8.0))
    if Ndim == 1:
        axs = [axs]

    axis_titles = [r"$\log(k_t)$", r"$\log(1/\Delta R)$"]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    for ii in range(Ndim):
        feature_idx = display_order[ii] if ii < len(display_order) else ii
        reference_counts, bin_edges = np.histogram(
            inputs[0][:, feature_idx],
            bins=hist1d_bins,
            range=hist1d_ranges[ii],
            density=False,
        )
        reference_total = np.sum(reference_counts)
        reference_density = reference_counts / reference_total if reference_total > 0 else reference_counts
        populated_reference = reference_counts >= max(1, int(min_reference_count))
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ratio_values = []

        for jj, arr in enumerate(inputs[1:], start=1):
            comparison_counts, _ = np.histogram(
                arr[:, feature_idx],
                bins=hist1d_bins,
                range=hist1d_ranges[ii],
                density=False,
            )
            comparison_total = np.sum(comparison_counts)
            comparison_density = comparison_counts / comparison_total if comparison_total > 0 else comparison_counts
            ratio = np.full_like(reference_density, np.nan, dtype=float)
            np.divide(
                reference_density - comparison_density,
                reference_density,
                out=ratio,
                where=populated_reference & (reference_density > 0.0),
            )
            ratio_values.append(ratio)
            label_idx = jj - 1
            axs[ii].step(
                centers,
                ratio,
                where="mid",
                linestyle=linestyles[label_idx % len(linestyles)],
                label=comparison_labels[label_idx] if label_idx < len(comparison_labels) else f"sample_{jj}",
            )

        axs[ii].axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
        axs[ii].set_title(axis_titles[ii] if ii < len(axis_titles) else f"feature_{ii}")
        axs[ii].set_xlabel("value")
        axs[ii].set_ylabel("Fractional diff" + (" (symlog)" if logy else ""))

        ymax = _symmetric_limit(ratio_values, fallback=1.0) * 1.15
        if max_abs_diff is not None and max_abs_diff > 0:
            ymax = min(ymax, float(max_abs_diff))
        if logy:
            axs[ii].set_yscale("symlog", linthresh=1.0)
            axs[ii].set_ylim(-ymax, ymax)
        else:
            axs[ii].set_ylim(-ymax, ymax)

    handles, legend_labels = axs[0].get_legend_handles_labels()
    note_lines = _plot_note_from_common(common_items, unavailable_notes)
    note_lines.append(
        f"Masked original bins with < {max(1, int(min_reference_count))} entries; y clipped at +/- {max_abs_diff:g}"
        if max_abs_diff is not None and max_abs_diff > 0
        else f"Masked original bins with < {max(1, int(min_reference_count))} entries"
    )
    bottom_edge = 0.20 if note_lines else 0.08
    fig.tight_layout(rect=[0.0, bottom_edge, 1.0, 1.0])
    if len(handles) > 0:
        for ax in axs:
            ax.legend(handles, legend_labels, loc="best", fontsize=8, framealpha=0.90)
    if note_lines:
        fig.text(
            0.02,
            0.02,
            _wrapped_note(note_lines, width=145),
            ha="left",
            va="bottom",
            fontsize=8,
        )

    if out_name is None:
        out_name = "hist1d_ratio_diff"

    fig.savefig(os.path.join(out_dir, out_name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, out_name + ".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_combined_1dhist(
    inputs,
    labels=None,
    out_dir="./Plots/",
    hist1d_ranges=None,
    hist1d_bins=30,
    logy=False,
    out_name=None,
    logy_floor_mode="clamped",
    unavailable_notes=None,
):
    """
    Plot overlaid 1D histograms for Lund features.

    Expected feature order:
      inputs[:, 0] = log(kt)
      inputs[:, 1] = log(1/deltaR)

    The top panel is log(kt), and the bottom panel is log(1/deltaR).
    This matches the plotting convention used in plot_ublund.py.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if labels is None:
        labels = [f"sample_{i}" for i in range(len(inputs))]
    diff_labels, common_items = _caption_comparison(labels, first_run_idx=1)
    plot_labels = ["original"]
    plot_labels.extend(diff_labels)

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    Ndim = inputs[0].shape[1]
    display_order = [0, 1] if Ndim >= 2 else list(range(Ndim))

    if hist1d_ranges is None:
        hist1d_ranges = _default_hist1d_ranges(inputs, display_order)

    fig, axs = plt.subplots(Ndim, 1, figsize=(8.0, 8.0))
    if Ndim == 1:
        axs = [axs]

    axis_titles = [r"$\log(k_t)$", r"$\log(1/\Delta R)$"]

    for ii in range(Ndim):
        # For ktdr plots, keep the top panel aligned with the Lund y-axis:
        # log(kt) first, then log(1/deltaR).
        feature_idx = display_order[ii] if ii < len(display_order) else ii

        all_hist_counts = []
        positive_hist_counts = []

        for jj, arr in enumerate(inputs):
            values = arr[:, feature_idx]

            hist_counts, _ = np.histogram(
                values,
                bins=hist1d_bins,
                range=hist1d_ranges[ii],
                density=True,
            )

            all_hist_counts.append(hist_counts)
            positive_hist_counts.extend(hist_counts[hist_counts > 0.0])

            axs[ii].hist(
                values,
                bins=hist1d_bins,
                range=hist1d_ranges[ii],
                histtype="step",
                density=True,
                linestyle=linestyles[jj % len(linestyles)],
                label=plot_labels[jj] if jj < len(plot_labels) else f"sample_{jj}",
            )

        axs[ii].set_title(axis_titles[ii] if ii < len(axis_titles) else f"feature_{ii}")
        axs[ii].set_xlabel("value")
        axs[ii].set_ylabel("Density (log scale)" if logy else "Density")

        ymax = max([np.max(h) for h in all_hist_counts if h.size > 0], default=1.0)

        if logy:
            axs[ii].set_yscale("log")
            if len(positive_hist_counts) > 0:
                min_positive = min(positive_hist_counts)
                max_positive = max(positive_hist_counts)

                if logy_floor_mode == "tail":
                    y_min = min_positive / 3.0
                else:
                    y_min = max(min_positive / 3.0, 1e-4)

                y_max = max_positive * 1.5
                if y_max <= y_min:
                    y_max = y_min * 10.0

                axs[ii].set_ylim(y_min, y_max)
            else:
                axs[ii].set_ylim(1e-4, 1.0)
        else:
            axs[ii].set_ylim(0.0, ymax * 1.15 if ymax > 0 else 1.0)

    handles, legend_labels = axs[0].get_legend_handles_labels()
    note_lines = _plot_note_from_common(common_items, unavailable_notes)
    has_caption = bool(note_lines)
    bottom_edge = 0.20 if has_caption else 0.08
    fig.tight_layout(rect=[0.0, bottom_edge, 1.0, 1.0])
    if len(handles) > 0:
        for ax in axs:
            ax.legend(
                handles,
                legend_labels,
                loc="best",
                fontsize=8,
                framealpha=0.90,
            )
    if note_lines:
        fig.text(
            0.02,
            0.02,
            _wrapped_note(note_lines, width=145),
            ha="left",
            va="bottom",
            fontsize=8,
        )

    if out_name is None:
        out_name = "hist1d_logy" if logy else "hist1d"

    fig.savefig(os.path.join(out_dir, out_name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, out_name + ".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_lund_ratio_diff(
    inputs,
    labels=["original","generated","predicted"],
    outdir="./Plots/",
    hist2d_xrange=None,
    hist2d_yrange=None,
    hist2d_bins=(20, 20),
    hist2d_shape=None,
    hist2d_layout=None,
    unavailable_notes=None,
    min_reference_count=5,
    max_abs_diff=1.0,
):
  if len(inputs) <= 1:
    return

  Ndim = inputs[0].shape[1]
  if Ndim < 2:
    return

  unavailable_notes = unavailable_notes or []
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  Ndiff = len(inputs) - 1
  Nplots = Ndiff + len(unavailable_notes)
  if Nplots == 0:
    return

  diff_labels, common_items = _caption_comparison(labels, first_run_idx=1)
  comparison_labels = diff_labels if len(diff_labels) > 0 else labels[1:]

  mins = np.zeros(Ndim)
  maxs = np.zeros(Ndim)
  for ii in range(Ndim):
    for jj in range(len(inputs)):
      mins[ii] = min(mins[ii], np.min(inputs[jj][:, ii]))
      maxs[ii] = max(maxs[ii], np.max(inputs[jj][:, ii]))

  mins[1] = -3
  maxs[0] = 8
  x_range = hist2d_xrange if hist2d_xrange is not None else [mins[1], maxs[1]]
  y_range = hist2d_yrange if hist2d_yrange is not None else [mins[0], maxs[0]]

  reference_counts, x_edges, y_edges = np.histogram2d(
      inputs[0][:, 1],
      inputs[0][:, 0],
      range=[x_range, y_range],
      bins=hist2d_bins,
      density=False,
  )
  reference_total = np.sum(reference_counts)
  reference_density = reference_counts / reference_total if reference_total > 0 else reference_counts
  populated_reference = reference_counts >= max(1, int(min_reference_count))

  ratio_maps = []
  for arr in inputs[1:]:
    comparison_counts, _, _ = np.histogram2d(
        arr[:, 1],
        arr[:, 0],
        range=[x_range, y_range],
        bins=hist2d_bins,
        density=False,
    )
    comparison_total = np.sum(comparison_counts)
    comparison_density = comparison_counts / comparison_total if comparison_total > 0 else comparison_counts
    ratio_map = np.full_like(reference_density, np.nan, dtype=float)
    np.divide(
        reference_density - comparison_density,
        reference_density,
        out=ratio_map,
        where=populated_reference & (reference_density > 0.0),
    )
    ratio_maps.append(ratio_map)

  if hist2d_shape is None:
    hist2d_shape = hist2d_layout
  nrows, ncols = resolve_hist2d_shape(Nplots, hist2d_shape)
  fig, axs = plt.subplots(
      nrows,
      ncols,
      figsize=(5.0 * ncols + 1.0, 4.4 * nrows + 0.6),
      squeeze=False,
  )
  flat_axs = axs.ravel()
  used_axs = flat_axs[:Nplots]

  vmax = _symmetric_limit(ratio_maps, fallback=1.0)
  if max_abs_diff is not None and max_abs_diff > 0:
    vmax = min(vmax, float(max_abs_diff))
  norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
  cmap = plt.get_cmap("coolwarm").copy()
  cmap.set_bad(color="#d9d9d9")
  last_image = None

  for jj, ratio_map in enumerate(ratio_maps):
    ax = used_axs[jj]
    last_image = ax.imshow(
        ratio_map.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(comparison_labels[jj] if jj < len(comparison_labels) else f"sample_{jj + 1}", pad=8, fontsize=10)
    ax.set_xlabel(r"$\log(1/\Delta R)$")
    ax.set_ylabel(r"$\log(k_t)$")

  for offset, (label, reason) in enumerate(unavailable_notes):
    ax = used_axs[Ndiff + offset]
    ax.set_title(label)
    ax.text(
        0.5,
        0.5,
        f"Plot unavailable\n{reason}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])

  for ax in flat_axs[Nplots:]:
    ax.set_visible(False)

  if last_image is not None:
    cbar = fig.colorbar(last_image, ax=used_axs.tolist(), fraction=0.022, pad=0.055, extend="both")
    cbar.set_label(r"$(original - generated) / original$")

  fig.suptitle("Lund Plane Fractional Difference", fontsize=14, y=0.985)
  note_lines = _plot_note_from_common(common_items, unavailable_notes)
  note_lines.append(
      f"Masked original bins with < {max(1, int(min_reference_count))} entries; color clipped at +/- {vmax:.3g}"
  )
  bottom_margin = 0.13 if note_lines else 0.09
  fig.subplots_adjust(left=0.08, right=0.88, bottom=bottom_margin, top=0.90, wspace=0.30, hspace=0.55)
  if note_lines:
    fig.text(
        0.08,
        0.025,
        _wrapped_note(note_lines, width=150),
        ha="left",
        va="bottom",
        fontsize=8,
    )

  name = "lund_ratio_diff"
  fig.savefig(os.path.join(outdir, name + ".png"), bbox_inches="tight")
  fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
  plt.close(fig)

def plot_combined_losses(run_infos, out_dir):
    diff_labels, common_items = _caption_comparison(
        ["original"] + [info.get("caption", info.get("checkpoint_path", f"run {i + 1}")) for i, info in enumerate(run_infos)],
        first_run_idx=1,
    )
    os.makedirs(out_dir, exist_ok=True)
    metric_names = set()
    for info in run_infos:
        for metric_name, curve in info.get("loss_curves_csv", {}).items():
            if curve is not None and len(curve) > 0:
                metric_names.add(metric_name)

    if "self_loss" in metric_names and "test_loss" in metric_names:
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        used_any = False
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        split_styles = {
            "Train": ("self_loss", "-"),
            "Val": ("test_loss", "--"),
        }
        for irun, info in enumerate(run_infos):
            label_base = diff_labels[irun] if irun < len(diff_labels) else f"run {irun + 1}"
            color = prop_cycle[irun % len(prop_cycle)] if prop_cycle else None
            for split_label, (metric_name, linestyle) in split_styles.items():
                curve = info.get("loss_curves_csv", {}).get(metric_name, None)
                if curve is None or len(curve) == 0:
                    continue
                y = np.asarray(curve, dtype=float)
                x = np.arange(1, len(y) + 1)
                finite = np.isfinite(y)
                truncated = False
                if np.any(~finite):
                    first_bad = int(np.argmax(~finite))
                    y = y[:first_bad]
                    x = x[:first_bad]
                    truncated = True
                if len(y) == 0 or not np.all(np.isfinite(y)):
                    continue

                label = f"{label_base} {split_label}"
                if truncated:
                    label = f"{label}\nvalid through ep {len(y)}"
                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    marker=markers[irun % len(markers)],
                    markersize=4,
                    markevery=max(len(x) // 12, 1),
                    linewidth=2.0,
                    alpha=0.9 if split_label == "Train" else 0.8,
                    label=label,
                )
                used_any = True

        if used_any:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Train and Validation Loss vs Epoch")
            ax.grid(True, alpha=0.35)
            legend_kwargs = dict(fontsize=8, framealpha=0.92)
            if len(run_infos) <= 4:
                ax.legend(loc="best", **legend_kwargs)
                bottom_edge = 0.14 if common_items else 0.06
            else:
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.18),
                    ncol=2,
                    **legend_kwargs,
                )
                bottom_edge = 0.30 if common_items else 0.24
            fig.tight_layout(rect=[0.0, bottom_edge, 1.0, 1.0])
            note_lines = _plot_note_from_common(common_items)
            if note_lines:
                fig.text(
                    0.02,
                    0.02,
                    _wrapped_note(note_lines, width=130),
                    ha="left",
                    va="bottom",
                    fontsize=8,
                )
            fig.savefig(os.path.join(out_dir, "loss_combined__train_val.png"), bbox_inches="tight")
            fig.savefig(os.path.join(out_dir, "loss_combined__train_val.pdf"), bbox_inches="tight")
            fig.savefig(os.path.join(out_dir, "loss_combined__training_and_val_loss.png"), bbox_inches="tight")
            fig.savefig(os.path.join(out_dir, "loss_combined__training_and_val_loss.pdf"), bbox_inches="tight")
        plt.close(fig)

    for metric_name in sorted(metric_names):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        used_any = False
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
        for irun, info in enumerate(run_infos):
            curve = info.get("loss_curves_csv", {}).get(metric_name, None)
            if curve is None or len(curve) == 0:
                continue
            y = np.asarray(curve, dtype=float)
            x = np.arange(1, len(y) + 1)
            finite = np.isfinite(y)
            truncated = False
            if np.any(~finite):
                first_bad = int(np.argmax(~finite))
                y = y[:first_bad]
                x = x[:first_bad]
                truncated = True
            if len(y) > 0 and np.all(np.isfinite(y)):
                label = diff_labels[irun] if irun < len(diff_labels) else f"run {irun + 1}"
                if truncated:
                    label = f"{label}\nvalid through ep {len(y)}"
                ax.plot(
                    x,
                    y,
                    color=prop_cycle[irun % len(prop_cycle)] if prop_cycle else None,
                    linestyle=linestyles[irun % len(linestyles)],
                    marker=markers[irun % len(markers)],
                    markersize=4,
                    markevery=max(len(x) // 12, 1),
                    linewidth=2.0,
                    alpha=0.9,
                    label=label,
                )
                used_any = True

        if not used_any:
            plt.close(fig)
            continue

        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL" if "nll" in metric_name.lower() else "Loss")
        ax.set_title(f"Loss vs Epoch ({metric_name})")
        ax.grid(True, alpha=0.35)
        legend_kwargs = dict(fontsize=8, framealpha=0.92)
        if len(run_infos) <= 5:
            ax.legend(loc="best", **legend_kwargs)
            bottom_edge = 0.14 if common_items else 0.06
        else:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=2,
                **legend_kwargs,
            )
            bottom_edge = 0.28 if common_items else 0.22
        fig.tight_layout(rect=[0.0, bottom_edge, 1.0, 1.0])
        note_lines = _plot_note_from_common(common_items)
        if note_lines:
            fig.text(
                0.02,
                0.02,
                _wrapped_note(note_lines, width=130),
                ha="left",
                va="bottom",
                fontsize=8,
            )
        fig.savefig(os.path.join(out_dir, f"loss_combined__{metric_name}.png"), bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, f"loss_combined__{metric_name}.pdf"), bbox_inches="tight")
        plt.close(fig)

def validate_unbinned_models(models, test_loader, args, labels=None, make_projection=False, unavailable_model_reasons=None,):
  if args.plot_max_batches is not None and args.plot_max_batches <= 0:
    raise ValueError("--plot-max-batches must be a positive integer")

  if labels is None:
    labels = ["original"]
    if len(models) == 1:
      labels.append("generated")
    else:
      labels.extend([f"generated_{ii}" for ii in range(len(models))])

  with torch.no_grad():
      original_chunks = []
      generated_chunks = [[] for _ in models]
      active_models = [True for _ in models]
      unavailable_reasons = [None for _ in models]

      device = "cpu"

      for imodel, model in enumerate(models):
        model.to(device)
        model.eval()
        forced_reason = None
        if unavailable_model_reasons is not None and imodel < len(unavailable_model_reasons):
          forced_reason = unavailable_model_reasons[imodel]
        if forced_reason:
          reason = forced_reason
          print(f"Generated plot unavailable for model {imodel}: {reason}.", flush=True)
          active_models[imodel] = False
          unavailable_reasons[imodel] = reason
        elif model_has_nonfinite_parameters(model):
          reason = "non-finite model parameters"
          print(f"Generated plot unavailable for model {imodel}: {reason}.", flush=True)
          active_models[imodel] = False
          unavailable_reasons[imodel] = reason

      printed_example = False
      for batch, X in enumerate(test_loader):
        if args.plot_max_batches is not None and batch >= args.plot_max_batches:
          break
        if batch>1000: break #FIXME

        #X = X.to(device)
        original_chunks.append(X)

        for imodel, model in enumerate(models):
          if not active_models[imodel]:
            continue

          try:
              if args.nf:
                  generated_seq = model.generate(out_dimensions=X.shape)
              elif args.diff:
                  generated_seq = model.generate(out_dimensions=X.shape)
              elif args.sde:
                  generated_seq = model.generate(out_dimensions=X.shape)
              elif args.cnf:
                  generated_seq = model.generate(out_dimensions=X.shape)
              elif args.fm:
                  generated_seq = model.generate(out_dimensions=X.shape)
              elif args.mdn:
                  generated_seq = model.generate(out_dimensions=X.shape)
              else:
                  generated_seq = model.generate(out_dimensions=X.shape)
          except RuntimeError as err:
            reason = f"generation failed: {err}"
            print(f"Generated plot unavailable for model {imodel}: {reason}", flush=True)
            active_models[imodel] = False
            generated_chunks[imodel] = []
            unavailable_reasons[imodel] = reason
            continue

          if not torch.isfinite(generated_seq).all():
            reason = "generated sequence contains nan/inf"
            print(f"Generated plot unavailable for model {imodel}: {reason}.", flush=True)
            active_models[imodel] = False
            generated_chunks[imodel] = []
            unavailable_reasons[imodel] = reason
            continue

          if args.mixed_loss:
            generated_seq[:, :, -1] = torch.sigmoid(generated_seq[:, :, -1])
            generated_seq[:, 0, -1] = X[:, 0, -1]

          if not printed_example:
            print("Input example")
            print(X[0])
            print("Generate example")
            starttime=time.time()
            print(generated_seq[0])
            print("Took %.2e ms to generate"%((time.time()-starttime)*1000), flush=True)
            printed_example = True

          generated_chunks[imodel].append(generated_seq)

      if len(original_chunks) == 0:
        raise ValueError("No validation batches were plotted")

      original = torch.cat(original_chunks)
      generated_list = [
          torch.cat(chunks)
          for chunks in generated_chunks
          if len(chunks) > 0
      ]
      active_labels = [labels[0]]
      active_labels.extend(
          labels[imodel + 1]
          for imodel, chunks in enumerate(generated_chunks)
          if len(chunks) > 0 and imodel + 1 < len(labels)
      )
      unavailable_notes = [
          (
              labels[imodel + 1] if imodel + 1 < len(labels) else f"generated_{imodel}",
              reason,
          )
          for imodel, reason in enumerate(unavailable_reasons)
          if reason is not None
      ]

      flat_original = original.flatten(0, 1).cpu().numpy()
      flat_generated_list = [g.flatten(0, 1).cpu().numpy() for g in generated_list]
      plot_inputs = [flat_original] + flat_generated_list

      if make_projection:
        projection_plot(
            plot_inputs,
            labels=active_labels,
            outdir=args.log_dir,
            unavailable_notes=unavailable_notes,
        )

      hist1d_ranges = None
      if args.hist1d_ranges is not None:
        if len(args.hist1d_ranges) != 4:
          raise ValueError("--hist1d-ranges should be: kt_min kt_max dr_min dr_max")
        hist1d_ranges = [
          [args.hist1d_ranges[0], args.hist1d_ranges[1]],
          [args.hist1d_ranges[2], args.hist1d_ranges[3]],
        ]

      plot_combined_1dhist(
          plot_inputs,
          labels=active_labels,
          out_dir=args.log_dir,
          hist1d_ranges=hist1d_ranges,
          hist1d_bins=args.hist1d_bins,
          logy=False,
          out_name="hist1d",
          unavailable_notes=unavailable_notes,
      )

      plot_combined_1dhist(
          plot_inputs,
          labels=active_labels,
          out_dir=args.log_dir,
          hist1d_ranges=hist1d_ranges,
          hist1d_bins=args.hist1d_bins,
          logy=True,
          out_name="hist1d_logy",
          unavailable_notes=unavailable_notes,
      )

      #Make plot
      if args.hist_ratio_diff and len(plot_inputs) > 1:
        plot_combined_1dhist_ratio_diff(
            plot_inputs,
            labels=active_labels,
            out_dir=args.log_dir,
            hist1d_ranges=hist1d_ranges,
            hist1d_bins=args.hist1d_bins,
            logy=False,
            out_name="hist1d_ratio_diff",
            unavailable_notes=unavailable_notes,
            min_reference_count=args.hist_ratio_min_count,
            max_abs_diff=args.hist_ratio_vmax,
        )

      if args.input_format == "ktdr":
        lund_inputs = plot_inputs
      else:
        lund_original = make_lundplane(original)
        lund_inputs = [lund_original.reshape(-1, lund_original.shape[-1])]
        for generated in generated_list:
          lund_generated = make_lundplane(generated)
          lund_inputs.append(lund_generated.reshape(-1, lund_generated.shape[-1]))

      lund_plot(
          lund_inputs,
          labels=active_labels,
          outdir=args.log_dir,
          hist2d_xrange=args.hist2d_xrange,
          hist2d_yrange=args.hist2d_yrange,
          hist2d_bins=args.hist2d_bins,
          hist2d_shape=args.hist2d_shape,
          unavailable_notes=unavailable_notes,
      )

      if args.hist_ratio_diff and len(lund_inputs) > 1:
        plot_lund_ratio_diff(
            lund_inputs,
            labels=active_labels,
            outdir=args.log_dir,
            hist2d_xrange=args.hist2d_xrange,
            hist2d_yrange=args.hist2d_yrange,
            hist2d_bins=args.hist2d_bins,
            hist2d_shape=args.hist2d_shape,
            unavailable_notes=unavailable_notes,
            min_reference_count=args.hist_ratio_min_count,
            max_abs_diff=args.hist_ratio_vmax,
        )

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

  fig,ax = plt.subplots(figsize=(6.0, 4.0))
  if n_epochs > 0:
    ax.plot(epochs, train_arr, marker="o", label="Train")
    ax.plot(epochs, test_arr, marker="o", label="Test")
  if truncated:
    ax.text(
      0.5,
      0.5,
      f"Loss history truncated at epoch {n_epochs}\nnext epoch contains nan/inf",
      ha="center",
      va="center",
      transform=ax.transAxes,
      fontsize=9,
    )
  ax.legend()
  ax.set_xlabel("Epoch")
  ax.set_ylabel("Loss")
  ax.set_yscale("log")
  ax.grid(True)
  fig.savefig(os.path.join(outdir,"loss_vs_epoch.png"))
  fig.savefig(os.path.join(outdir,"loss_vs_epoch.pdf"))
  fig.savefig(os.path.join(outdir,"loss_train_val_vs_epoch.png"))
  fig.savefig(os.path.join(outdir,"loss_train_val_vs_epoch.pdf"))
  plt.close(fig)

  save_loss_csv(
    epoch_losses=loss_train,
    loss_curves={"test_loss": loss_test, **(loss_curves or {})},
    out_dir=outdir,
  )

if __name__ == "__main__":
    pass
