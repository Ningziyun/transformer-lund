#!/usr/bin/env python3

import numpy as np
import os,sys
import ROOT           
import math
from tqdm import tqdm
#from argparse import ArgumentParser

from helpers_train import *

import torch
from torchinfo import summary
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
torch.multiprocessing.set_sharing_strategy("file_system")
from torchviz import make_dot

import matplotlib.pyplot as plt

def quickLundPlot(inputs,labels=["original","generated","predicted"]):

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
    fig.colorbar(pos[3],ax=axs)

    name="lund"
    fig.savefig(name+".png")
    fig.savefig(name+".pdf")
    plt.close(fig)

  #1D plots
  fig, axs = plt.subplots(Ndim,1,figsize=(8.0,8.0))
  if Ndim==1: axs=[axs]
  for ii in range(Ndim):
    for jj in range(Nin):
      axs[ii].hist(inputs[jj][:,ii],bins=20,range=[mins[ii],maxs[ii]],histtype="step",density=True,linestyle=linestyles[jj],label=labels[jj])
    if ii==0: axs[0].legend()

  name="projection"
  fig.savefig(name+".png")
  fig.savefig(name+".pdf")
  plt.close(fig)

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
    #inputs=np.array([self.DR[index]])
    #inputs=np.array([self.kt[index]])
    if self.add_stop:
      inputs=np.concatenate([inputs,[self.stop[index]]],axis=0)
    
    self.data=torch.transpose(torch.tensor(inputs),0,1)

    return self.data

  def __len__(self):
    return len(self.DR)

def get_loaders(batch_size=256, num_workers=1, shuffle=True,):
  train_dataset = input_dataset("inputFiles/discretized/qcd_lund_cut_lundTree_kt_deltaR_train.h5")
  train_loader = DataLoader( train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,)
  test_dataset = input_dataset("inputFiles/discretized/qcd_lund_cut_lundTree_kt_deltaR_val.h5")
  test_loader = DataLoader( test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,)
  return train_loader,test_loader

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


##########################################
class test_modelMDN(test_model):
  def __init__(self, input_dim, n_mix=25, embed_dim=128, num_heads=1, num_layers=2, ff_dim=128):
      super(test_modelMDN, self).__init__(input_dim, embed_dim=128, num_heads=1, num_layers=2, ff_dim=128)

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

##########################################
class test_modelQuantile(test_model):
  def __init__(self, input_dim, embed_dim=128, num_heads=1, num_layers=2, ff_dim=128):
      super(test_model3, self).__init__(input_dim, embed_dim=128, num_heads=1, num_layers=2, ff_dim=128)

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

if __name__ == "__main__":
    args = parse_input()
    save_arguments(args)
    print(f"Logging to {args.log_dir}")
    set_seeds(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    num_features = args.num_features # Origin was 3(Step-3)
    num_bins = tuple(args.num_bins)

    # load and preprocess data
    print(f"Loading training set")

    train_loader,test_loader=get_loaders()

    X=next(iter(train_loader))
    print("Input shape,",X.shape)

    doMDN=False
    doMixedLoss=False

    # construct model
    if args.contin:
        model = load_model(log_dir=args.model_path)
        print("Loaded model")
    else:
        #model=test_modelNN(X.shape[1],X.shape[2])
        if doMDN:
          model=test_modelMDN(X.shape[2])
        else:
          model=test_model(X.shape[2])

    summary(model,input_data=[X[:,:-1,:]], col_names=["input_size", "output_size", "num_params","params_percent","mult_adds","trainable"])
    print("Output shape,",model(X[:,:-1,:]).shape)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if not doMDN:
      loss_fn = nn.MSELoss(reduction='none')   # regression next-step prediction
      if doMixedLoss:
        #loss_fn2 = nn.CrossEntropyLoss() #expects logits
        loss_fn2 = nn.BCEWithLogitsLoss(reduction='none') #expects logits
        sigmoid=nn.Sigmoid()
    else:
      loss_fn=mdn_loss

    best_loss=1e6
    patience_counter=0
    patience=3
    loss_test=[]
    loss_train=[]
    epochs=10
    for t in range(epochs):
      for batch, X in enumerate(train_loader):
          X = X.to(device)
          optimizer.zero_grad()

          inputs = X[:, :-1, :]   # all but last
          targets = X[:, 1:, :]   # all but first
          #targets=inputs
          #inputs=X.clone().detach()
          #inputs[:,-1,:]=0
          #targets=X

          pred = model(inputs)       # (batch, seq_len-1, feature_dim)
          #print(inputs.shape,targets.shape,pred.shape)

          if not doMDN:
            if doMixedLoss:
              lambd=1
              #print(loss_fn(pred[:,:,:-1],targets[:,:,:-1]).shape,loss_fn2(pred[:,:,-1],targets[:,:,-1]).shape)
              loss= loss_fn(pred[:,:,:-1],targets[:,:,:-1]).mean(dim=-1)+lambd*loss_fn2(pred[:,:,-1],targets[:,:,-1]) #mixed loss
            else:
              loss = loss_fn(pred, targets) #whole loss

            #mask = torch.ones(inputs.shape,device=device,dtype=torch.bool) #mask and loss dimension [Nbatch,NConst,Nfeatures]
            #mask = inputs[:,:,0]>-1
            #loss = loss[mask].mean()
            loss=loss.mean()
          else:
            loss = loss_fn(pred, targets)
            loss=loss.mean()

          #FIXME quantile regression
          #quantiles = [0.1, 0.5, 0.9]
          #loss = quantile_loss(pred, targets, quantiles)

          #FIXME playing around
          #loss=loss_fn(pred[:,0],targets[:,0])

          #loss = loss_fn(pred[:, -1, :], targets[:, -1, :]) #last element loss
          #loss = loss_fn(pred, targets[:, -1, :])
          #loss = loss_fn(pred[:,:,0],targets[:,:,0]) #only first feature regression

          loss.backward()
          optimizer.step()

          if batch % 100 == 0:
            print(f"batch: {batch} loss:{loss.item()}")

      if doMixedLoss:
        pred[:,:,-1]=sigmoid(pred[:,:,-1])

      print(f"epoch: {t} loss={loss.item()}")

      #print("inputs",inputs[0])
      #print("targets",targets[0])
      #print("pred",pred[0])

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
        pred= model(X)
        generated_seq = model.generate(start, steps=X.shape[1]-1)

        if doMixedLoss:
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

      quickLundPlot([original.numpy(),generated.numpy()])
