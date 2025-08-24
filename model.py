import numpy as np
import torch
from torch.nn import (
    Module,
    ModuleList,
    Embedding,
    Linear,
    TransformerEncoderLayer,
    CrossEntropyLoss,
    LayerNorm,
    Dropout,
)


class EmbeddingProductHead(Module):
    def __init__(self, hidden_dim=256, num_features=3, num_bins=(41, 41, 41)):
        super(EmbeddingProductHead, self).__init__()
        assert num_features == 3
        self.num_features = num_features
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self.combined_bins = int(np.sum(num_bins))
        self.linear = Linear(hidden_dim, self.combined_bins * hidden_dim)
        self.act = torch.nn.Softplus()
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, emb):
        batch_size, seq_len, _ = emb.shape
        bin_emb = self.act(self.linear(emb))
        bin_emb = bin_emb.view(batch_size, seq_len, self.combined_bins, self.hidden_dim)
        bin_emb_x, bin_emb_y, bin_emb_z = torch.split(bin_emb, self.num_bins, dim=2)

        bin_emb_xy = bin_emb_x.unsqueeze(2) * bin_emb_y.unsqueeze(3)
        bin_emb_xy = bin_emb_xy.view(batch_size, seq_len, -1, self.hidden_dim)

        logits = bin_emb_xy @ bin_emb_z.transpose(2, 3)
        logits = self.logit_scale.exp() * logits.view(batch_size, seq_len, -1)
        return logits


class JetTransformerClassifier(Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=10,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
    ):
        super(JetTransformerClassifier, self).__init__()
        '''
        Step 1 Delete
        self.num_features = num_features
        self.dropout = dropout

        # learn embedding for each bin of each feature dim
        self.feature_embeddings = ModuleList(
            [
                Embedding(embedding_dim=hidden_dim, num_embeddings=num_bins[l])
                for l in range(num_features)
            ]
        )
        '''
        # Step 1 Add
        self.num_features = num_features
        self.num_bins = num_bins            # keep per-feature bin counts
        self.dropout = dropout

        # Reserve one PAD slot per feature (padding_idx equals the original n_bins)
        self.pad_idx = tuple(b for b in num_bins)

        # learn embedding for each bin of each feature dim (+1 for PAD)
        self.feature_embeddings = ModuleList(
            [
                Embedding(
                    embedding_dim=hidden_dim,
                    num_embeddings=num_bins[l] + 1,  # +1 for PAD slot
                    padding_idx=num_bins[l]          # PAD index is the extra slot
                )
                for l in range(num_features)
            ]
        )

        # Step 1 End

        # build transformer layers
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                    norm_first=True,
                    dropout=dropout,
                )
                for l in range(num_layers)
            ]
        )

        self.out_norm = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)

        # output projection and loss criterion
        self.flat = torch.nn.Flatten()
        self.out = Linear(hidden_dim * 100, 1)
        self.criterion = torch.nn.functional.binary_cross_entropy_with_logits

    def forward(self, x, padding_mask):
        # construct causal mask to restrict attention to preceding elements
        seq_len = x.shape[1]
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=x.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)
        padding_mask = ~padding_mask

        # Step 1 add
        # --- sanitize indices and map invalid values to PAD (detect NaNs before casting) ---
        x_f = x  # keep float view for NaN detection if x is float
        x = x.long()

        emb = None
        for i in range(self.num_features):
            idx = x[:, :, i]

            # detect NaNs from the original float tensor (if available)
            nan_mask = torch.isnan(x_f[:, :, i].float()) if x_f.dtype.is_floating_point else torch.zeros_like(idx, dtype=torch.bool)

            # invalid if < 0, >= n_bins[i], or NaN
            bad = (idx < 0) | (idx >= self.num_bins[i]) | nan_mask

            # map invalid values to PAD slot
            pad_val = torch.full_like(idx, self.pad_idx[i])
            idx = torch.where(bad, pad_val, idx)

            e = self.feature_embeddings[i](idx)
            emb = e if emb is None else emb + e
        # --- end sanitize ---
        # Step 1 End

        # apply transformer layer
        for layer in self.layers:
            emb = layer(
                src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask
            )

        emb = self.out_norm(emb)
        emb = self.dropout(emb)
        emb = self.flat(emb)
        out = self.out(emb)
        return out

    def loss(self, logits, true_bin):
        loss = self.criterion(logits, true_bin)
        return loss


class JetTransformer(Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=10,
        num_heads=4,
        num_features=3,
        num_bins=(41, 31, 31),
        dropout=0.1,
        output="linear",
        classifier=False,
        tanh=False,
        end_token=False,
    ):
        super(JetTransformer, self).__init__()
        self.num_features = num_features
        self.num_bins = num_bins
        self.dropout = dropout
        self.total_bins = int(np.prod(num_bins))
        if end_token:
            self.total_bins += 1
        self.classifier = classifier
        self.tanh = tanh
        print(f"Bins: {self.total_bins}")
        '''
        Step1-delete
        # learn embedding for each bin of each feature dim
        self.feature_embeddings = ModuleList(
            [
                Embedding(embedding_dim=hidden_dim, num_embeddings=num_bins[l])
                for l in range(num_features)
            ]
        )
        '''
        # Step-1 Adding
        # Reserve one PAD slot per feature (padding_idx = original n_bins)
        self.pad_idx = tuple(b for b in num_bins)  # PAD index for each feature

        # Learn embedding for each feature (+1 for PAD). We keep logits size = prod(num_bins),
        # i.e., PAD is never a class; PAD is only for inputs/targets masking.
        self.feature_embeddings = ModuleList(
            [
                Embedding(
                    embedding_dim=hidden_dim,
                    num_embeddings=num_bins[l] + 1,  # +1 for PAD slot
                    padding_idx=num_bins[l]          # PAD index is the extra slot
                )
                for l in range(num_features)
            ]
        )
        # Step-1 End


        # build transformer layers
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                    norm_first=True,
                    dropout=dropout,
                )
                for l in range(num_layers)
            ]
        )

        self.out_norm = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)

        # output projection and loss criterion
        if output == "linear":
            self.out_proj = Linear(hidden_dim, self.total_bins)
        else:
            self.out_proj = EmbeddingProductHead(hidden_dim, num_features, num_bins)
        '''
        Step-2 Delete
        self.criterion = CrossEntropyLoss()
        '''
        # Step-2 Adding
        # Ignore invalid / PAD targets in the loss (we do not have a PAD class in the logits)
        self.ignore_index = -100
        self.criterion = CrossEntropyLoss(ignore_index=self.ignore_index)
        # Step-2 End

    def forward(self, x, padding_mask):
        # construct causal mask to restrict attention to preceding elements
        seq_len = x.shape[1]
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=x.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)
        padding_mask = ~padding_mask
        '''
        Step-1 Delete
        # project x to initial embedding
        x[x < 0] = 0
        emb = self.feature_embeddings[0](x[:, :, 0])
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](x[:, :, i])
        '''
        # Step-1 Adding
        # --- sanitize indices and map invalid values to PAD before embedding ---
        # NOTE: PAD is the (original) upper bound num_bins[i], thanks to +1 embeddings above.
        x_f = x  # keep float view for NaN detection if upstream passed floats
        x = x.long()  # Embedding requires int64 indices

        emb = None
        for i in range(self.num_features):
            idx = x[:, :, i]

            # invalid if < 0, >= n_bins[i], or NaN (detected on the float view)
            nan_mask = torch.isnan(x_f[:, :, i].float()) if x_f.dtype.is_floating_point else torch.zeros_like(idx, dtype=torch.bool)
            bad = (idx < 0) | (idx >= self.num_bins[i]) | nan_mask

            # map all invalid values to the PAD slot
            pad_val = torch.full_like(idx, self.pad_idx[i])
            idx = torch.where(bad, pad_val, idx)

            e = self.feature_embeddings[i](idx)
            emb = e if emb is None else emb + e
        # --- end sanitize ---
        # Step-1 End

        # apply transformer layer
        for layer in self.layers:
            emb = layer(
                src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask
            )

        emb = self.out_norm(emb)
        emb = self.dropout(emb)

        # project final embedding to logits (not normalized with softmax)
        logits = self.out_proj(emb)
        if self.tanh:
            return 13 * torch.tanh(0.1 * logits)
        else:
            return logits

    '''
    Step-2 Delete
    def loss(self, logits, true_bin):
        # ignore final logits
        logits = logits[:, :-1].reshape(-1, self.total_bins)

        # shift target bins to right
        true_bin = true_bin[:, 1:].flatten()

        loss = self.criterion(logits, true_bin)
        return loss
    '''
    # Step-2 Adding
    def loss(self, logits, true_bin):
        # logits: (B, T, total_bins); we ignore the last step (teacher forcing next-token)
        logits = logits[:, :-1].reshape(-1, self.total_bins)

        # shift targets to the right to align with logits
        true_bin = true_bin[:, 1:].reshape(-1)

        # --- map invalid targets to ignore_index ---
        # Any target < 0, >= total_bins, or NaN should be ignored (padded steps, bad triples)
        if true_bin.dtype.is_floating_point:
            nan_mask = torch.isnan(true_bin)
            true_bin = true_bin.long()
        else:
            nan_mask = torch.zeros_like(true_bin, dtype=torch.bool)

        bad = (true_bin < 0) | (true_bin >= self.total_bins) | nan_mask
        true_bin = torch.where(
            bad,
            torch.full_like(true_bin, self.ignore_index),  # send to ignore_index
            true_bin,
        )
        # --- end map ---

        return self.criterion(logits, true_bin)
    # Step-2 End


    def probability(
        self,
        logits,
        padding_mask,
        true_bin,
        perplexity=False,
        logarithmic=False,
        topk=False,
    ):
        batch_size, padded_seq_len, num_bin = logits.shape
        seq_len = padding_mask.long().sum(dim=1)

        # ignore final logits
        logits = logits[:, :-1]
        probs = torch.softmax(logits, dim=-1)

        if topk:
            vals, idx = torch.topk(probs, topk, dim=-1, sorted=False)
            probs = torch.zeros_like(probs, device=probs.device)
            probs[
                torch.arange(probs.shape[0])[:, None, None],
                torch.arange(probs.shape[1])[None, :, None],
                idx,
            ] = vals

            probs = probs / probs.sum(dim=-1, keepdim=True)

        probs = probs.reshape(-1, self.total_bins)

        # shift target bins to right
        true_bin = true_bin[:, 1:].flatten()

        # select probs of true bins
        sel_idx = torch.arange(probs.shape[0], dtype=torch.long, device=probs.device)
        probs = probs[sel_idx, true_bin].view(batch_size, padded_seq_len - 1)
        probs[~padding_mask[:, 1:]] = 1.0
        if perplexity:
            probs = probs ** (1 / seq_len.float().view(-1, 1))

        if logarithmic:
            probs = torch.log(probs).sum(dim=1)
        else:
            probs = probs.prod(dim=1)
        return probs

    def sample_old(self, starts, device, len_seq, trunc=None):
        def select_idx():
            # Select bin at random according to probabilities
            rand = torch.rand((len(jets), 1), device=device)
            preds_cum = torch.cumsum(preds, -1)
            preds_cum[:, -1] += 0.01  # If rand = 1, sort it to the last bin
            idx = torch.searchsorted(preds_cum, rand).squeeze(1)
            return idx

        if not trunc is None and trunc >= 1:
            trunc = torch.tensor(trunc, dtype=torch.long)

        jets = -torch.ones((len(starts), len_seq, 3), dtype=torch.long, device=device)
        true_bins = torch.zeros((len(starts), len_seq), dtype=torch.long, device=device)

        # Set start bins and constituents
        num_prior_bins = torch.cumprod(torch.tensor([1, 41, 31]), -1).to(device)
        bins = (starts * num_prior_bins.reshape(1, 1, 3)).sum(axis=2)
        true_bins[:, 0] = bins
        jets[:, 0] = starts
        padding_mask = jets[:, :, 0] != -1

        self.eval()
        finished = torch.ones(len(starts)) != 1
        with torch.no_grad():
            for particle in range(len_seq - 1):
                if all(finished):
                    break
                # Get probabilities for the next particles
                preds = self.forward(jets, padding_mask)[:, particle]
                preds = torch.nn.functional.softmax(preds[:, :], dim=-1)

                # Remove low probs
                if not trunc is None:
                    if trunc < 1:
                        preds = torch.where(
                            preds < trunc, torch.zeros(1, device=device), preds
                        )
                    else:
                        preds, indices = torch.topk(preds, trunc, -1, sorted=False)

                preds = preds / torch.sum(preds, -1, keepdim=True)

                idx = select_idx()
                if not trunc is None and trunc >= 1:
                    idx = indices[torch.arange(len(indices)), idx]
                finished[idx == self.total_bins] = True

                # Get tuple from found bin and set next particle properties
                true_bins[~finished, particle + 1] = idx[~finished]
                bins = self.idx_to_bins(idx[~finished])
                for ind, tmp_bin in enumerate(bins):
                    jets[~finished, particle + 1, ind] = tmp_bin

                padding_mask[~finished, particle + 1] = True
        return jets, true_bins

    def sample(self, starts, device, len_seq, trunc=None):
        def select_idx():
            # Select bin at random according to probabilities
            rand = torch.rand((len(jets), 1), device=device)
            preds_cum = torch.cumsum(preds, -1)
            preds_cum[:, -1] += 0.01  # If rand = 1, sort it to the last bin
            idx = torch.searchsorted(preds_cum, rand).squeeze(1)
            return idx

        if not trunc is None and trunc >= 1:
            trunc = torch.tensor(trunc, dtype=torch.long)

        jets = -torch.ones((len(starts), 1, 3), dtype=torch.long, device=device)
        true_bins = torch.zeros((len(starts), 1), dtype=torch.long, device=device)

        # Set start bins and constituents
        num_prior_bins = torch.cumprod(torch.tensor([1, self.num_bins[0], self.num_bins[1]]), -1).to(device)
        bins = (starts * num_prior_bins.reshape(1, 1, 3)).sum(axis=2)
        true_bins[:, 0] = bins
        jets[:, 0] = starts
        padding_mask = jets[:, :, 0] != -1

        self.eval()
        finished = torch.ones(len(starts), device=device) != 1
        with torch.no_grad():
            for particle in range(len_seq - 1):
                if all(finished):
                    break
                # Get probabilities for the next particles
                preds = self.forward(jets, padding_mask)[:, particle]
                preds = torch.nn.functional.softmax(preds[:, :], dim=-1)

                # Remove low probs
                if not trunc is None:
                    if trunc < 1:
                        preds = torch.where(
                            preds < trunc, torch.zeros(1, device=device), preds
                        )
                    else:
                        preds, indices = torch.topk(preds, trunc, -1, sorted=False)

                preds = preds / torch.sum(preds, -1, keepdim=True)

                idx = select_idx()
                if not trunc is None and trunc >= 1:
                    idx = indices[torch.arange(len(indices)), idx]
                finished[idx == self.total_bins] = True

                # Get tuple from found bin and set next particle properties
                true_bins = torch.concat((true_bins, idx.view(-1, 1)), dim=1)
                bins = self.idx_to_bins(idx)
                bins[finished] = 0
                jets = torch.concat((jets, bins.view(-1, 1, self.num_features)), dim=1)
                padding_mask = torch.concat((padding_mask, ~finished.view(-1, 1)), dim=1)

        return jets, true_bins


    def idx_to_bins(self, x):
        pT = x % self.num_bins[0]

        # Step 1 Adding
        device = x.device
        nb0 = torch.tensor(self.num_bins[0], device=device)
        nb01 = torch.prod(torch.tensor(self.num_bins[:2], device=device))

        eta = torch.div((x - pT), nb0, rounding_mode="trunc") % torch.div(nb01, nb0, rounding_mode="trunc")
        phi = torch.div((x - pT - nb0 * eta), nb01, rounding_mode="trunc")
        # Step 1 End
        return torch.stack((pT, eta, phi), dim=-1)


class CNNclass(Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = torch.nn.Sequential(
            # Input = 1 x 30 x 30, Output = 32 x 30 x 30
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            # Input = 32 x 30 x 30, Output = 32 x 15 x 15
            torch.nn.MaxPool2d(kernel_size=2),
            # Input = 32 x 15 x 15, Output = 64 x 15 x 15
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            # Input = 64 x 15 x 15, Output = 64 x 7 x 7
            torch.nn.MaxPool2d(kernel_size=2),
            # Input = 64 x 7 x 7, Output = 64 x 7 x 7
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            # Input = 64 x 7 x 7, Output = 64 x 3 x 3
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 3 * 3, 512),
            torch.nn.PReLU(),
            torch.nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(x, y)


class ParticleNet(Module):
    pass
