"""
VQ-VAE tokenizer for jet constituents, feature-agnostic version.

The model works directly on whatever per-constituent features are stored in the input .h5
(4-vectors, relative vectors, Lund-plane (kt, dR), ...), with the pre-processing already
applied there. No kinematic transform or normalisation is done inside the model; inverting
the pre-processing is left to the user (helpers.undo_preprocess).

    forward(x)              x: (M, F) REAL constituents only (select with X[~mask])
    encode(X, mask)         (B, N, F) -> (B, N, Q) codes, -1 where padded
    decode(codes)           (B, N, Q) -> (B, N, F) features, pad_value where padded
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """Codebook updated with exponential moving averages (van den Oord et al. 2017,
    Razavi et al. 2019) with dead-code restarts. Codes are initialised from data on the
    first training steps (all codes start 'dead')."""

    def __init__(self, num_codes, code_dim, decay=0.99, beta=0.25, eps=1e-5, dead_threshold=0.03):
        super().__init__()
        self.K, self.D = num_codes, code_dim
        self.decay, self.beta, self.eps, self.dead_threshold = decay, beta, eps, dead_threshold
        self.register_buffer("embed", torch.randn(num_codes, code_dim) * 0.1)
        self.register_buffer("embed_avg", self.embed.clone())
        self.register_buffer("cluster_size", torch.zeros(num_codes))

    def lookup(self, codes):
        return F.embedding(codes, self.embed)

    def quantize(self, z):  # z: (M, D)
        d = (z.pow(2).sum(1, keepdim=True) - 2 * z @ self.embed.t() + self.embed.pow(2).sum(1)[None])
        return d.argmin(1)

    def forward(self, z):  # z: (M, D)
        z = z.float()
        codes = self.quantize(z)
        zq = self.lookup(codes)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(codes, self.K).type(z.dtype)
                n = onehot.sum(0)
                s = onehot.t() @ z
                self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(s, alpha=1 - self.decay)
                total = self.cluster_size.sum()
                cs = (self.cluster_size + self.eps) / (total + self.K * self.eps) * total
                self.embed.copy_(self.embed_avg / cs.unsqueeze(1))

                dead = self.cluster_size < self.dead_threshold
                n_dead = int(dead.sum())
                if n_dead > 0:  # restart unused codes on random encoder outputs
                    idx = torch.randint(0, z.shape[0], (n_dead,), device=z.device)
                    new = z[idx] + 1e-3 * torch.randn_like(z[idx])
                    self.embed[dead] = new
                    self.embed_avg[dead] = new
                    self.cluster_size[dead] = 1.0

        commit = self.beta * F.mse_loss(z, zq.detach())
        zq_st = z + (zq - z).detach()  # straight-through estimator

        probs = torch.bincount(codes, minlength=self.K).float() / codes.numel()
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
        return zq_st, codes, commit, perplexity


class ResidualVQ(nn.Module):
    """Q stacked quantisers, each quantising the residual of the previous ones.
    Q=1 is a plain VQ-VAE; Q>1 gives finer resolution at the cost of Q tokens/constituent."""

    def __init__(self, num_levels, num_codes, code_dim, **kw):
        super().__init__()
        self.levels = nn.ModuleList([VectorQuantizerEMA(num_codes, code_dim, **kw) for _ in range(num_levels)])

    def forward(self, z):
        residual, zq_total = z, torch.zeros_like(z)
        codes, losses, perps = [], 0.0, []
        for vq in self.levels:
            zq, c, loss, p = vq(residual)
            residual = residual - zq.detach()
            zq_total = zq_total + zq
            codes.append(c)
            losses = losses + loss
            perps.append(p)
        return zq_total, torch.stack(codes, -1), losses, torch.stack(perps)

    def lookup(self, codes):  # codes (M, Q)
        return sum(vq.lookup(codes[:, q]) for q, vq in enumerate(self.levels))


def mlp(din, dh, dout, n_hidden=2):
    layers, d = [], din
    for _ in range(n_hidden):
        layers += [nn.Linear(d, dh), nn.SiLU()]
        d = dh
    layers.append(nn.Linear(d, dout))
    return nn.Sequential(*layers)


class ConstituentVQVAE(nn.Module):
    """Per-constituent tokenizer: F features -> Q codebook indices -> F features.

    input_format is stored as metadata only (e.g. "4vec", "relvec", "ktdr"); n_features must
    match the data. pad_value is what decode() writes into padded slots, so the output has the
    same padding convention as the (pre-processed) input file.
    """

    def __init__(self, num_codes=512, num_levels=2, code_dim=16, hidden=256, input_format=None,
                 n_features=4, pad_value=-1.0, decay=0.99, beta=0.25):
        super().__init__()
        self.config = dict(num_codes=num_codes, num_levels=num_levels, code_dim=code_dim, hidden=hidden,
                           input_format=input_format, n_features=n_features, pad_value=pad_value,
                           decay=decay, beta=beta)
        self.K, self.Q, self.F = num_codes, num_levels, n_features
        self.input_format, self.pad_value = input_format, pad_value
        self.encoder = mlp(n_features, hidden, code_dim, n_hidden=3)
        self.decoder = mlp(code_dim, hidden, n_features, n_hidden=3)
        self.vq = ResidualVQ(num_levels, num_codes, code_dim, decay=decay, beta=beta)

    # -- training forward on flat real constituents -------------------
    def forward(self, x):
        if x.dim() != 2 or x.shape[-1] != self.F:
            raise ValueError(f"ConstituentVQVAE.forward expects flat real constituents of shape (M, {self.F}), "
                             f"got {tuple(x.shape)}. Select them with X[~mask] (mask is True for padding).")
        x = x.float()
        z = self.encoder(x)
        zq, codes, commit, perp = self.vq(z)
        rec = self.decoder(zq)
        recon = F.mse_loss(rec, x)
        return rec, codes, recon, commit, perp

    # -- tokenizer API --------------------------------------------------
    def _residuals(self, z):
        residual = z
        for vq in self.vq.levels:
            yield vq, residual
            residual = residual - vq.lookup(vq.quantize(residual))

    @torch.no_grad()
    def encode(self, X, mask):
        """(B, N, F) features, (B, N) mask (True = padded) -> (B, N, Q) codes, -1 where padded."""
        codes = torch.full((*mask.shape, self.Q), -1, dtype=torch.long, device=X.device)
        real = ~mask
        if real.any():
            z = self.encoder(X[real].float())
            codes[real] = torch.stack([vq.quantize(r) for vq, r in self._residuals(z)], -1)
        return codes

    @torch.no_grad()
    def decode(self, codes):
        """(B, N, Q) codes (-1 = padded) -> (B, N, F) features, pad_value where padded."""
        mask = codes[..., 0] < 0
        out = torch.full((*mask.shape, self.F), float(self.pad_value), device=codes.device)
        if (~mask).any():
            out[~mask] = self.decoder(self.vq.lookup(codes[~mask]))
        return out
