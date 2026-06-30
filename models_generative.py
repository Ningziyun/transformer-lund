import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim.lr_scheduler import LambdaLR
torch.multiprocessing.set_sharing_strategy("file_system")
from torchviz import make_dot

import normflows as nf
from torchdiffeq import odeint_adjoint as odeint

# ---------------------------------------------------------------------
# Simple auto-regressive NN
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Autoregressive trasnformer model
# ---------------------------------------------------------------------
class model_autoregressive_transformer(nn.Module):
  '''
  Auto-regressive trasnformer, learns next element prediction p(x_i|x_{<i}). During training takes x[0:-1] and learns to predict x[1:] via a transformer. For generation always needs a seed x[0], then can recursively generate the rest of the elements
  '''
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
  '''
  Exactly like the previous auto-regressive model, but models the next prediction as a gaussian mixture model as opposed to exact value. Seems to avoid mode collapse
  '''
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

# ---------------------------------------------------------------------
# CNF (Continous Normalizing Flow)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Flow matching
# ---------------------------------------------------------------------
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

    def mse_loss(self,x):
        return self.fm.loss(x)

    @torch.no_grad()
    def generate(self,out_dimensions):

        samples=self.fm.generate( batch_size=out_dimensions[0])

        return samples.view(out_dimensions)

# ---------------------------------------------------------------------
# Custom vanilla Normlazing flows, better to use nflows package
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Normalizing flows
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Diffusion model like DDPM/DDIM
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Score based stochastic differntial equation
# ---------------------------------------------------------------------
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
