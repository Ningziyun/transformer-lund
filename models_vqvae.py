"""
VQ-VAE tokenizer for jet constituents

No kinematic transform or normalisation is done inside the model; inverting
the pre-processing is left to the user
    forward(x)              x: (M, F) REAL constituents only (select with X[~mask])
    encode(X, mask)         (B, N, F) -> (B, N, Q) codes, -1 where padded
    decode(codes)           (B, N, Q) -> (B, N, F) features, pad_value where padded
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Interfaces for saving/loading
# ---------------------------------------------------------------------
def load_tokenizer_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = ConstituentVQVAE(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["train_args"]

def save_tokenizer_checkpoint(model, args, history, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config,
                "state_dict": model.state_dict(),
                "training_history": history,
                "train_args": vars(args)},
                path)

# ---------------------------------------------------------------------
# Actual Residual VQ-VAE
# ---------------------------------------------------------------------
def MLP(dim_in, dim_embed, dim_out, n_layers=2):
    '''
    Basic MLP as backbone of the encoder/decoder
    '''
    layers, d = [], dim_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, dim_embed), nn.SiLU()]
        d = dim_embed
    layers.append(nn.Linear(d, dim_out))
    return nn.Sequential(*layers)

class VectorQuantizerEMA(nn.Module):
    """
    Codebook updated with exponential moving averages (van den Oord et al. 2017,
    Razavi et al. 2019) with dead-code restarts. Codes are initialised from data on the
    first training steps (all codes start 'dead').

    Takes in M latent vectors
    """

    def __init__(self, num_codes, code_dim, decay=0.99, beta=0.25, eps=1e-5, dead_threshold=0.03):
        super().__init__()
        self.K, self.D = num_codes, code_dim

        #Since using EMA, actually store the embedding as buffer so not updated by Adam, but instead here
        self.decay, self.beta, self.eps, self.dead_threshold = decay, beta, eps, dead_threshold
        self.register_buffer("embed", torch.randn(num_codes, code_dim) * 0.1)
        self.register_buffer("embed_avg", self.embed.clone())
        self.register_buffer("cluster_size", torch.zeros(num_codes))

    def lookup(self, codes):
        """
        Find the embedding z_q from q
        Effectively like one-hot embeedings then bias-free MLP using the weights from self.embed 
        """
        return F.embedding(codes, self.embed)

    def quantize(self, z):  # z: (M, D)
        '''
        Calculate |z-e_k|^2 but instead of looping over each latent vector and each codebook entry, calculate all pairwise squared distances using matrix operations
        |z-e_k|^2=|z|^2 -2 * z^T * e^k + |e^k|^2
        returns the dimension with lowest value
        '''
        d = (z.pow(2).sum(1, keepdim=True) - 2 * z @ self.embed.t() + self.embed.pow(2).sum(1)[None])
        return d.argmin(1)

    def forward(self, z):  # z: (M, D)

        #Find the nearest code k, and it's vector z_q
        z = z.float()
        codes = self.quantize(z)
        zq = self.lookup(codes)

        #For running the EMA training
        if self.training:
            with torch.no_grad():
                #one-hot embedding of the M vectors in K codes (M, K)
                onehot = F.one_hot(codes, self.K).type(z.dtype)
                nk = onehot.sum(0) #Cluster size: Count occurance of the code (K,)
                skd = onehot.t() @ z #Embedding average: Sum of input z mapped to this (K, D)

                #Caluclate the rolling EMA values of these
                #N_k^t=gamma*N_k^{t-1}+(1+gamma)*n_k
                #S_k^t=gamma*S_k^{t-1}+(1+gamma)*s_k (actully doing S_{k,d}^t for later easy)
                self.cluster_size.mul_(self.decay).add_(nk, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(skd, alpha=1 - self.decay)

                #Now apply the optimization iteration. The update is technically e_k=S_k/N_k
                #Instead use smoothed version to protect against divide by zero N_k=(N_k+eps)/(sum_j N_j+eps*K) * sum_j N_j
                total = self.cluster_size.sum()
                cs = (self.cluster_size + self.eps) / (total + self.K * self.eps) * total
                self.embed.copy_(self.embed_avg / cs.unsqueeze(1))

                #Count number of dead codes (aka those now used alot)
                dead = self.cluster_size < self.dead_threshold
                n_dead = int(dead.sum())
                if n_dead > 0:
                    #Sample a random input point and add some gaussian noise to it
                    idx = torch.randint(0, z.shape[0], (n_dead,), device=z.device)
                    new = z[idx] + 1e-3 * torch.randn_like(z[idx])

                    #Replace with this and reset the EMA
                    self.embed[dead] = new
                    self.embed_avg[dead] = new
                    self.cluster_size[dead] = 1.0

        #The commit loss beta*|z-sg(z_q)|^2
        commit_loss = self.beta * F.mse_loss(z, zq.detach())

        # straight-through estimator: return zq but gives dz_q/dz=I
        zq_st = z + (zq - z).detach()

        #Perplexity P=2^H(p)=2^(-sum p*log_2(p))
        probs = torch.bincount(codes, minlength=self.K).float() / codes.numel()
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())

        #Return
        return zq_st, codes, commit_loss, perplexity

class ResidualVQ(nn.Module):
    """
    Q stacked quantisers, each quantising the residual of the previous ones.
    Q=1 is a plain VQ-VAE; Q>1 gives finer resolution at the cost of Q tokens/constituent.

    Map latent z to nearest code z_q. For >1 layers return the stack of codes [Q, D]
    Since doing residual VQ will do this N times and the new vector is actually the total vector z_{q,total}= sum_i^N z_q^i
    """

    def __init__(self, num_levels, num_codes, code_dim, **kw):
        super().__init__()
        self.levels = nn.ModuleList([VectorQuantizerEMA(num_codes, code_dim, **kw) for _ in range(num_levels)])

    def forward(self, z):
        #First "residual" is just the encoder input, set total zq to zero to sum in
        residual = z
        zq_total=torch.zeros_like(z)

        #Loop over the levels
        codes, total_commit_loss, perps = [], 0.0, []
        for vq in self.levels:
            #Calculate the VQ
            zq, code, commit_loss, p = vq(residual)

            #Sum the commit loss and embed-vector across layers
            total_commit_loss = total_commit_loss + commit_loss
            zq_total = zq_total + zq

            #Store info
            codes.append(code)
            perps.append(p)

            #Shift so next layer input is the residual difference. Detach so the layers across gradients sepertae
            residual = residual - zq.detach()

        #Return info, importantly the codes are stacked across layers
        return zq_total, torch.stack(codes, -1), total_commit_loss, torch.stack(perps)

    def lookup(self, codes):  # codes (M, Q)
        '''
        Returns z_q from given codes. Return [M, D]
        '''
        return sum(vq.lookup(codes[:, q]) for q, vq in enumerate(self.levels))

class ConstituentVQVAE(nn.Module):
    """
    Per-constituent tokenizer: F features -> Q codebook indices -> F features.

    input_format is stored as metadata only (e.g. "4vec", "relvec", "ktdr"); n_features must
    match the data. pad_value is what decode() writes into padded slots, so the output has the
    same padding convention as the (pre-processed) input file.
    """

    def __init__(self, num_codes=512, num_levels=2, code_dim=16, latent_dim=256, input_format=None, n_features=4, pad_value=-1.0, decay=0.99, beta=0.25):
        super().__init__()
        #TODO maybe better to use kwargs above to store config dict easily, but this works
        self.config = dict(num_codes=num_codes, num_levels=num_levels, code_dim=code_dim, latent_dim=latent_dim,
                           input_format=input_format, n_features=n_features, pad_value=pad_value,
                           decay=decay, beta=beta)

        self.K, self.Q, self.F = num_codes, num_levels, n_features
        self.input_format, self.pad_value = input_format, pad_value

        #The encoder/decoder model
        self.encoder = MLP(n_features, latent_dim, code_dim, n_layers=3)
        self.decoder = MLP(code_dim, latent_dim, n_features, n_layers=3)

        #The residual stack which is num_layers deep
        self.vq = ResidualVQ(num_levels, num_codes, code_dim, decay=decay, beta=beta)

    #training forward on flat real constituents, dimension (M, F) not (B, C, F)
    def forward(self, x):
        if x.dim() != 2 or x.shape[-1] != self.F:
            raise ValueError(f"ConstituentVQVAE.forward expects flat real constituents of shape (M, {self.F}), "
                             f"got {tuple(x.shape)}. Select them with X[~mask] (mask is True for padding).")

        #The encoder
        x = x.float()
        z = self.encoder(x)

        #The vector-quantization
        zq, codes, commit_loss, perp = self.vq(z)

        #The decoder and reconstruction loss
        y = self.decoder(zq)
        reco_loss = F.mse_loss(y, x)

        #Return a bunch of info: decoder output, codes, reconstuctio loss, commit loss, and perplexity
        return y, codes, reco_loss, commit_loss, perp

    @torch.no_grad()
    def encode(self, X, mask):
        """
        X:    (B, C, F) input features
        mask: (B, C), True = padding

        Returns:
            codes: (B, C, Q), with -1 for padded constituents
        """

        #Set a return tensor to code padding=-1 [B,C,Q]
        codes = torch.full( (*mask.shape, self.Q), -1, dtype=torch.long, device=X.device,)

        # Select only real constituents
        real = ~mask
        if not real.any():
            return codes

        # Encode input into latent space
        z = self.encoder(X[real].float())

        #Loop over the layers and do residual VQ-VAE
        residual = z
        for q, vq in enumerate(self.vq.levels):
            # Find the nearest code for the current residual
            code = vq.quantize(residual)

            # Store this level's token
            codes[..., q][real] = code

            # Subtract its embedding to obtain the next residual
            zq = vq.lookup(code)
            residual = residual - zq

        return codes

    @torch.no_grad()
    def decode(self, codes):
        """(B, N, Q) codes (-1 = padded) -> (B, N, F) features, pad_value where padded."""

        #Make mask and output tensor set to pad value
        mask = codes[..., 0] < 0
        out = torch.full((*mask.shape, self.F), float(self.pad_value), device=codes.device)

        #Find the z_q and decode that
        if (~mask).any():
            out[~mask] = self.decoder(self.vq.lookup(codes[~mask]))
        return out
