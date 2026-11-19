import os,sys
import re
import numpy as np
import math
import h5py

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import fastjet
import ljpHelpers
import helpers_lundtree

import models_generative

# ---------------------------------------------------------------------
# Make the lundplane
# ---------------------------------------------------------------------
def make_lundplane(input_vec, pad_length=20):

  #Assume input is dimensions [Nevents, Nconstituents, 4-vecs]
  jetDef10 = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0, fastjet.E_scheme)
  #jetDefCA = fastjet.JetDefinition1Param(fastjet.cambridge_algorithm, 10.0)

  lund_plane=[]
  for ii in range(input_vec.shape[0]):

    # Convert the constituent information into a format usable for fastjet (PseudoJet objects)
    constituents = [ fastjet.PseudoJet( float(px), float(py), float(pz), float(E),) for E, px, py, pz in input_vec[ii] if E>0 ] 

    # Run the jet clustering on the jet constituents using the anti-kt algorithm
    cs_akt = fastjet.ClusterSequence(constituents, jetDef10)
    inclusiveJets10 = fastjet.sorted_by_pt(cs_akt.inclusive_jets(25.))

    # Skip if inclusiveJets10 is empty
    if not inclusiveJets10: continue

    # Get Lund plane declusterings
    lundPlane = ljpHelpers.jet_declusterings(inclusiveJets10[0])
    lp_points = np.full((pad_length, 2), -1.0, dtype=np.float32) #Note padded out to -1 already
    for kk in range(min(len(lundPlane),pad_length)):
      if (lundPlane[kk].delta_R > 0 and lundPlane[kk].z > 0):
        dr_val = math.log(1.0 / lundPlane[kk].delta_R)
        kt_val = math.log(lundPlane[kk].kt)
        lp_points[kk]=[kt_val,dr_val]

    # Free C++ memory
    constituents.clear()
    del cs_akt
    del inclusiveJets10
    del lundPlane

    #push back and clean-up
    lund_plane.append(lp_points)
    del lp_points

  #pad the length
  lund_plane=np.asarray(lund_plane)
  return lund_plane

# ---------------------------------------------------------------------
# Make the full lund-tree
# ---------------------------------------------------------------------
def make_lundtree(input_vec, Nconst_max=100):
  '''
  Full Lund tree of the leading anti-kt jet, reclustered with C/A, see helpers_lundtree for the conventions.
  Input is [Nevents, Nconstituents, 4-vecs] like make_lundplane, only the first Nconst_max constituents are used.
  Returns the model tokens [Nevents, 2*Nconst_max-1, len(helpers_lundtree.TOKEN_NAMES)] padded with -1, and the
  [Nevents, 4] jet [E, px, py, pz] the tree belongs to (its pt/eta/phi is the conditioning used for reconstruction).
  '''

  #Physical node features and split flags, padded out to the largest possible tree
  Nnodes = 2*Nconst_max-1
  features = np.zeros([input_vec.shape[0], Nnodes, len(helpers_lundtree.SPLIT_FEATURE_NAMES)], dtype=np.float32)
  splits = np.full([input_vec.shape[0], Nnodes], -1, dtype=np.int8)
  for ii in range(input_vec.shape[0]):
    nodes = ljpHelpers.constituents_to_lund_tree(input_vec[ii][:Nconst_max])
    features[ii, :len(nodes)] = [node[0] for node in nodes]
    splits[ii, :len(nodes)] = [int(node[1]) for node in nodes]

  #Encode into the model space and add the context of the next node, both need a walk over each tree
  prepocessed, context = helpers_lundtree.preprocess__trees(features, splits)
  tokens = np.concatenate([prepocessed, context, splits[..., None]], axis=-1).astype(np.float32)
  tokens[splits == -1] = -1 #pad
  return tokens

# ---------------------------------------------------------------------
# Converters between representations
# ---------------------------------------------------------------------
def make_absolute_constituents(relative_vec, jet_vec):
    '''
    Convert relative const vectors to absolute ones
    '''

    # Jet quantities
    jet_pt = jet_vec[:, 0]
    jet_eta = jet_vec[:, 1]
    jet_phi = jet_vec[:, 2]

    # Relative constituent quantities
    pt_frac = relative_vec[:, :, 0]
    delta_eta = relative_vec[:, :, 1]
    delta_phi = relative_vec[:, :, 2]

    # Recover constituent lab-frame coordinates
    const_pt = pt_frac * jet_pt[:, None]
    const_phi = delta_phi + jet_phi[:, None]
    const_eta = delta_eta + jet_eta[:, None]

    # 4-vector asumming massless
    E = const_pt * np.cosh(const_eta)
    px = const_pt * np.cos(const_phi)
    py = const_pt * np.sin(const_phi)
    pz = const_pt * np.sinh(const_eta)

    #Apply pad mask
    mask = pt_frac<0
    E[mask]=-1
    px[mask]=-1
    py[mask]=-1
    pz[mask]=-1

    return np.stack([E, px, py, pz], axis=-1)

def jet_PtEtaPhiM(jet):
    """(.., 4) E,px,py,pz -> pt, eta, phi, m, E"""
    E, px, py, pz = np.split(jet,4,-1)
    pt = np.hypot(px, py)
    eta = np.arcsinh(pz / np.maximum(pt, 1e-12))
    phi = np.arctan2(py, px)
    m = np.sqrt(np.maximum(E ** 2 - (px ** 2 + py ** 2 + pz ** 2), 0.0))
    return pt, eta, phi, m, E

# ---------------------------------------------------------------------
# Functions to help with loading in/out config data
# ---------------------------------------------------------------------
def as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)

def as_int(value, default=None):
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

def as_float(value, default=None):
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

def config_get(args_or_dict, key, default=None):
    '''
    Check if exists in arparser or normal dictionary
    '''
    if isinstance(args_or_dict, dict):
        return args_or_dict.get(key, default)
    return getattr(args_or_dict, key, default)

# ---------------------------------------------------------------------
# Data loaders / kinematics
# ---------------------------------------------------------------------
# All formats are streamed from the .h5 files written by lunddata.py: only a shuffle buffer is held in RAM,
# never the whole file. Memory per DataLoader worker is ~2 x block_size x blocks_per_buffer jets
# (~0.3-0.6 GB with the defaults, for any format), independent of the file size.
#
#   * the file is split into contiguous blocks of `block_size` jets
#   * each epoch the block order is shuffled; each worker reads `blocks_per_buffer` random blocks into a
#     buffer, shuffles the jets inside it and yields ready-made batches (DataLoader uses batch_size=None)
#   * lunddata.py writes the JetClass ROOT files one after the other (one process class per region of the
#     file), so mixing many random blocks per buffer is what keeps each batch class-mixed
#
# Batches have the same layout as the old in-memory datasets: X, or [X, jet], [X, mask], [X, jet, mask].
# dataset.X / .mask / .jet_data still exist for old code, as disk-backed views: slicing them
# (dataset.X[:10000], dataset.jet_data[idx]) only reads those rows. len(dataset) is the number of
# BATCHES (what DataLoader needs); the number of jets is dataset.n_jets.

def _format_spec(input_format):
    '''(h5 group, feature datasets, default pad value, truncate to num_constituents, mask rule, jet conditioning)'''
    if input_format=="ktdr":
        return "lundplane", ["kt","dr"], -1.0, True, "all", "full"
    elif input_format=="4vec":
        return "constituents", ["E","PX","PY","PZ"], -10.0, True, "all", "full"
    elif input_format=="relvec":
        return "relative_constituents", ["ptfrac","deta","dphi"], -10.0, True, "all", "full"
    elif input_format=="lundtree":
        #NOTE: a tree can't be truncated without breaking it, its size is set in lunddata.py. Only the split flag
        #(last token) is -1 for padding, -1 is a valid feature. The tree fixes the mass and multiplicity, so the
        #conditioning is only the [pt, eta, phi] of the anti-kt R=1.0 jet it was built from
        return "lund_tree", list(helpers_lundtree.TOKEN_NAMES), -1.0, False, "last", "ptetaphi"
    raise ValueError(f"No known input format {input_format}")

class block_stream_dataset(torch.utils.data.IterableDataset):
    '''
    Generic block-shuffled streaming over the rows (jets) of an .h5 file. Subclasses implement
        _read(f, blocks, **kw) -> tuple of numpy arrays (or None) with one row per jet, for the (start, stop) ranges
        _pack(*arrays)         -> what one batch yields
    and call super().__init__ with the number of jets and the bytes per jet (used to size the shuffle buffer).
    '''
    BUFFER_BYTES = 160 * 1024**2 #target size of one shuffle buffer (RAM per worker is ~2-3x this)

    def __init__(self, file_path, n_jets, bytes_per_jet, batch_size=256, shuffle=True, block_size=None,
                 blocks_per_buffer=32, drop_last=False, max_jets=None, num_workers=0):
        super(block_stream_dataset, self).__init__()
        self.file_path=file_path
        self.batch_size=int(batch_size)
        self.shuffle=shuffle
        self.drop_last=drop_last #True for training: exact batch count per epoch (e.g. for OneCycleLR)
        self.num_workers=num_workers
        self.n_jets = n_jets if max_jets is None else min(n_jets, int(max_jets))

        #Shuffle buffer = blocks_per_buffer random blocks of block_size contiguous jets. By default the block size
        #is chosen so a buffer is ~BUFFER_BYTES whatever the jet size (lund trees are ~3x bigger than relvec jets)
        self.blocks_per_buffer = int(blocks_per_buffer)
        if block_size is None:
            block_size = int(np.clip(self.BUFFER_BYTES // (self.blocks_per_buffer * max(bytes_per_jet, 1)), 256, 16384))
        self.block_size = int(block_size)
        self.blocks = [(s, min(s+self.block_size, self.n_jets)) for s in range(0, self.n_jets, self.block_size)]

    def _read(self, f, blocks, **kw):
        raise NotImplementedError

    def _pack(self, *arrays):
        raise NotImplementedError

    def __len__(self):
        '''
        number of BATCHES per epoch, an upper bound: each worker drops (drop_last) or yields a partial
        last batch, so with several workers there can be up to num_workers-1 batches fewer
        '''
        if self.drop_last:
            return self.n_jets // self.batch_size
        return math.ceil(self.n_jets / self.batch_size) + max(self.num_workers - 1, 0)

    def _plan(self):
        '''block order for this epoch (shared by all workers) and this worker's share of it'''
        info = torch.utils.data.get_worker_info()
        if info is None:
            base, wid, nw = int(torch.randint(0, 2**62, ()).item()), 0, 1
        else: #info.seed = base_seed + id, base_seed changes every epoch and follows torch.manual_seed
            base, wid, nw = info.seed - info.id, info.id, info.num_workers
        if self.shuffle:
            order = np.random.default_rng(base).permutation(len(self.blocks))
        else:
            order = np.arange(len(self.blocks))
        return order[wid::nw], np.random.default_rng(base + 1 + wid)

    def __iter__(self):
        '''
        Without shuffling and with at most one worker the jets come out in file order. With several workers
        each worker reads its own blocks and their batches arrive interleaved.
        '''
        my_blocks, rng = self._plan()
        bs = self.batch_size
        take = lambda arrays, sel: tuple(None if a is None else a[sel] for a in arrays)
        n_buffers = max(1, math.ceil(len(my_blocks) / self.blocks_per_buffer))
        with h5py.File(self.file_path, "r") as f:
            carry = None
            for buffer_blocks in np.array_split(my_blocks, n_buffers): #equal-sized buffers, so the last one is mixed too
                if len(buffer_blocks)==0:
                    continue
                blocks = [self.blocks[b] for b in buffer_blocks]
                if not self.shuffle:
                    blocks.sort()
                arrays = self._read(f, blocks)
                if carry is not None:
                    arrays = tuple(None if a is None else np.concatenate([c, a]) for c, a in zip(carry, arrays))
                n = len(arrays[0])
                if self.shuffle:
                    arrays = take(arrays, rng.permutation(n))
                n_full = n // bs * bs
                for s in range(0, n_full, bs):
                    yield self._pack(*take(arrays, slice(s, s+bs)))
                carry = take(arrays, slice(n_full, None))
                del arrays
            if carry is not None and len(carry[0]) and not self.drop_last:
                yield self._pack(*carry)

    def _slices(self, n_jets, n_slices=128):
        '''n_slices short row ranges spread evenly over the file, n_jets rows in total'''
        n_jets = min(int(n_jets), self.n_jets)
        n_slices = max(1, min(n_slices, n_jets))
        length = math.ceil(n_jets / n_slices)
        if length * n_slices >= self.n_jets:
            return [(0, n_jets)]
        starts = np.linspace(0, self.n_jets - length, n_slices).astype(int)
        return [(int(s), int(s) + length) for s in starts]

    def sample(self, n_jets, n_slices=128, **kw):
        '''
        Load n_jets jets for validation/diagnostics, from n_slices short slices spread evenly over the file
        (the first n jets would all come from the first ROOT file = one process class).
        '''
        n_jets = min(int(n_jets), self.n_jets)
        with h5py.File(self.file_path, "r") as f:
            arrays = self._read(f, self._slices(n_jets, n_slices), **kw)
        return tuple(None if a is None else a[:n_jets] for a in arrays)

    def read_rows(self, rows, **kw):
        '''read arbitrary jet indices (int array, any order, repeats allowed) from disk, returned in that order'''
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        rows = np.where(rows < 0, rows + self.n_jets, rows)
        if len(rows) and (rows.min() < 0 or rows.max() >= self.n_jets):
            raise IndexError(f"jet index out of range for {self.n_jets} jets")
        uniq, inverse = np.unique(rows, return_inverse=True)
        #group sorted indices into contiguous runs, then read each run in one go
        breaks = np.flatnonzero(np.diff(uniq) != 1) + 1
        runs = [(int(r[0]), int(r[-1])+1) for r in np.split(uniq, breaks) if len(r)]
        with h5py.File(self.file_path, "r") as f:
            arrays = self._read(f, runs, **kw)
        return tuple(None if a is None else a[inverse] for a in arrays)

    def stream_all(self, **kw):
        '''iterate over the whole file in file order as tuples of numpy arrays (one block at a time), e.g. for statistics'''
        with h5py.File(self.file_path, "r") as f:
            for block in self.blocks:
                yield self._read(f, [block], **kw)

class jet_stream_dataset(block_stream_dataset):
    '''Jets in one of the lunddata.py formats (ktdr, 4vec, relvec, lundtree), see the notes above.'''

    def __init__(self, file_path, input_format, NConstituents=20, batch_size=256, shuffle=True, add_mask=False,
                 add_conditional=False, preprocess=None, pad_value=None, block_size=None, blocks_per_buffer=32,
                 drop_last=False, max_jets=None, num_workers=0):
        self.input_format=input_format
        self.group, self.keys, default_pad, truncate, self.mask_rule, self.jet_rule = _format_spec(input_format)
        self.pad_value=float(default_pad if pad_value is None else pad_value)
        self.n_features=len(self.keys)
        self.add_mask=add_mask
        self.add_conditional=add_conditional
        self.preprocess=preprocess #NOTE: pre-processing is applied in lunddata.py, stored only as metadata

        #Only metadata is read here, the file is opened inside each worker (h5py handles don't survive a fork)
        with h5py.File(file_path, "r") as f:
            shape = f[self.group][self.keys[0]].shape
            self.has_jet = "jet" in f
        if add_conditional and not self.has_jet:
            raise KeyError(f"{file_path} has no 'jet' group, needed for the conditioning")
        self.width = min(shape[1], int(NConstituents)) if truncate and NConstituents else shape[1]

        super(jet_stream_dataset, self).__init__(file_path, shape[0], self.width*self.n_features*4, batch_size=batch_size,
            shuffle=shuffle, block_size=block_size, blocks_per_buffer=blocks_per_buffer, drop_last=drop_last,
            max_jets=max_jets, num_workers=num_workers)

    def _mask(self, X):
        #Mask which stores if padded. True/False=Padded/unpadded
        if self.mask_rule=="last":
            return X[..., -1]==np.float32(self.pad_value)
        return np.all(X==np.float32(self.pad_value), axis=-1)

    def _jet(self, E4, mask):
        pt, eta, phi, m, _ = jet_PtEtaPhiM(E4)
        if self.jet_rule=="ptetaphi":
            return np.concatenate([pt, eta, phi], axis=-1).astype(np.float32)
        n_real = (~mask).sum(axis=1).reshape(-1,1)
        return np.concatenate([pt, eta, phi, m, n_real], axis=-1).astype(np.float32)

    def _read(self, f, blocks, need_jet=None):
        '''read a list of (start, stop) row ranges -> X [n, C, F] float32, mask [n, C] bool, jet [n, J] float32 or None'''
        need_jet = self.add_conditional if need_jet is None else need_jet
        n = sum(e-s for s, e in blocks)
        X = np.empty((n, self.width, self.n_features), dtype=np.float32)
        E4 = np.empty((n, 4), dtype=np.float64) if need_jet else None
        grp = f[self.group]
        pos = 0
        for s, e in blocks:
            for j, k in enumerate(self.keys):
                X[pos:pos+e-s, :, j] = grp[k][s:e, :self.width]
            if E4 is not None:
                for j, k in enumerate(["E","PX","PY","PZ"]):
                    E4[pos:pos+e-s, j] = f["jet"][k][s:e]
            pos += e-s
        mask = self._mask(X)
        jet = self._jet(E4, mask) if E4 is not None else None
        return X, mask, jet

    def _pack(self, X, mask, jet):
        X = torch.from_numpy(np.ascontiguousarray(X))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        jet = None if jet is None else torch.from_numpy(np.ascontiguousarray(jet))
        if self.add_mask and self.add_conditional:
            return [X, jet, mask]
        elif self.add_mask:
            return [X, mask]
        elif self.add_conditional:
            return [X, jet]
        return X

    def sample(self, n_jets, n_slices=128):
        '''-> X [n, C, F] float32, mask [n, C] bool, jet [n, J] float32 (always computed if the file has a jet group)'''
        return super(jet_stream_dataset, self).sample(n_jets, n_slices, need_jet=self.has_jet)

    @property
    def X(self):
        return _disk_view(self, "X")

    @property
    def mask(self):
        return _disk_view(self, "mask")

    @property
    def jet_data(self):
        return _disk_view(self, "jet")

class _disk_view:
    '''
    Array-like stand-in for the old in-memory dataset.X / .mask / .jet_data. Indexing reads only the requested
    jets from disk, e.g. dataset.X[:10000], dataset.jet_data[idx], dataset.mask[5]. np.asarray(view) on the
    full X or mask loads the whole file (the old behaviour) and warns, so avoid it on big files.
    '''
    _WARN_BYTES = 2 * 1024**3

    def __init__(self, dataset, field):
        self.ds, self.field = dataset, field
        if field=="X":
            self.shape = (dataset.n_jets, dataset.width, dataset.n_features)
            self.dtype = np.dtype(np.float32)
        elif field=="mask":
            self.shape = (dataset.n_jets, dataset.width)
            self.dtype = np.dtype(bool)
        else:
            self.shape = (dataset.n_jets, 3 if dataset.jet_rule=="ptetaphi" else 5)
            self.dtype = np.dtype(np.float32)
        self.ndim = len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        rows, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        if isinstance(rows, (int, np.integer)):
            out = self._rows(np.array([rows]))[0]
            return out[rest] if rest else out
        else:
            if isinstance(rows, slice):
                rows = np.arange(*rows.indices(self.shape[0]))
            elif isinstance(rows, torch.Tensor):
                rows = rows.cpu().numpy()
            rows = np.asarray(rows)
            if rows.dtype == bool:
                rows = np.flatnonzero(rows)
            out = self._rows(rows)
        return out[(slice(None),) + rest] if rest else out #rows already selected, apply the rest to the other axes

    def _rows(self, rows):
        X, mask, jet = self.ds.read_rows(rows, need_jet=(self.field=="jet"))
        return {"X": X, "mask": mask, "jet": jet}[self.field]

    def __array__(self, dtype=None, copy=None):
        nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        if nbytes > self._WARN_BYTES:
            print(f"WARNING: loading the full dataset.{self.field} of {self.ds.file_path} into RAM "
                  f"({nbytes/1024**3:.1f} GB); slice it instead, e.g. dataset.{self.field}[:N]", flush=True)
        out = self[:]
        return out if dtype is None else out.astype(dtype)

def get_loader(filename, args, train=True):
    '''
    Handles all the data-loader configurations. train=True drops each worker's last partial batch
    (fixed number of optimiser steps per epoch); the test loader keeps every jet
    '''

    #Interface for differnt argparser setups between vqvae and generative training
    add_mask=config_get(args, "multi_loss", False)
    add_conditional=config_get(args, "conditional", False)
    if config_get(args, "codebook_size"):
        add_mask=True
        add_conditional=True

    input_format=config_get(args, "input_format","")
    if input_format=="relvec" or input_format=="lundtree":
        add_conditional=True #the jet is needed to reconstruct the constituents

    batch_size=config_get(args, "batch_size", 0)
    num_workers=config_get(args, "num_workers", 0)
    shuffle=config_get(args, "shuffle", True)
    dataset = jet_stream_dataset(
        filename, input_format,
        NConstituents=config_get(args, "num_constituents", 0),
        batch_size=batch_size,
        shuffle=shuffle,
        add_mask=add_mask,
        add_conditional=add_conditional,
        preprocess=config_get(args, "preprocess", False),
        pad_value=config_get(args, "pad_value", None),
        block_size=config_get(args, "block_size", None),
        blocks_per_buffer=config_get(args, "blocks_per_buffer", 32) or 32,
        max_jets=config_get(args, "max_jets", None),
        drop_last=train and shuffle,
        num_workers=num_workers,
    )

    #batch_size=None: the dataset already yields whole batches
    use_cuda = str(config_get(args, "device", "") or "").startswith("cuda") or torch.cuda.is_available()
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=use_cuda)
    return loader

def get_loaders(args):
    '''
    Calls above
    '''

    train_loader=get_loader(args.train_file, args)
    test_loader=get_loader(args.test_file, args, train=False)

    return train_loader, test_loader

# ---------------------------------------------------------------------
# Data format handling
# ---------------------------------------------------------------------
def format_input_args(args):

    func_args={"unpack_mask":False, "unpack_jet": False, "device": args.device}

    input_format=config_get(args,"input_format","")
    multi_loss=config_get(args,"multi_loss",False)
    conditional=config_get(args,"conditional",False)

    needs_jet = input_format=="relvec" or input_format=="lundtree" #the loader always returns the jet for these
    if (needs_jet and multi_loss) or (conditional and multi_loss):
        func_args["unpack_mask"]=True
        func_args["unpack_jet"]=True
    elif needs_jet or conditional:
        func_args["unpack_jet"]=True
    elif multi_loss:
        func_args["unpack_mask"]=True

    return func_args

def format_input(X, unpack_mask, unpack_jet, device):

    #Format the output for each case
    if unpack_mask and unpack_jet:
        X, jet, mask = X
    elif unpack_mask:
        X, mask = X
        jet=None
    elif unpack_jet:
        X, jet = X
        mask=None
    else:
        mask=None
        jet=None

    #Set to devices
    X = X.to(device)
    if mask is not None: mask=mask.to(device)
    if jet is not None: jet=jet.to(device)

    return X, mask, jet

# ---------------------------------------------------------------------
# Macros to help with training
# ---------------------------------------------------------------------
def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def model_has_nonfinite_parameters(model):
    for tensor in model.state_dict().values():
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
            return True
    return False

# ---------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------
def preprocess_mean_std(input_format):
    #Predefined feature for standardizing
    if input_format=="ktdr":
        dr_mean,dr_std=2.786, 1.517
        kt_mean,kt_std=-0.015, 1.534
        return [[kt_mean,kt_std],[dr_mean,dr_std]]

    elif input_format=="4vec":
        e_mean,e_std=26.006, 60.301
        px_mean,px_std=-0.002, 30.131
        py_mean,py_std=0.011, 30.174
        pz_mean,pz_std=-0.001, 49.941
        return [[e_mean,e_std],[px_mean,px_std],[py_mean,py_std],[pz_mean,pz_std]]

    elif input_format=="relvec":
        ptfrac_mean,ptfrac_std=0.029, 0.061
        deta_mean,deta_std=0.000, 0.200
        dphi_mean,dphi_std=0.000, 0.202
        return [[ptfrac_mean,ptfrac_std],[deta_mean,deta_std],[dphi_mean,dphi_std]]

        #ptfrac_mean,ptfrac_std=4.531,1.298
    return None

def preprocess_log_mean_std(input_format):
    #Predefined feature for standardizing
    if input_format=="4vec":
        e_mean,e_std=2.417,1.173
        px_mean,px_std=0.000,2.019
        py_mean,py_std=0.000,2.021
        pz_mean,pz_std=0.000,2.213
        return [[e_mean,e_std],[px_mean,px_std],[py_mean,py_std],[pz_mean,pz_std]]

    elif input_format=="relvec":
        #ptfrac_mean,ptfrac_std=4.531,1.298 #-log(x)
        ptfrac_mean,ptfrac_std=0.027,0.052
        deta_mean,deta_std=0.000, 0.200
        dphi_mean,dphi_std=0.000, 0.202
        return [[ptfrac_mean,ptfrac_std],[deta_mean,deta_std],[dphi_mean,dphi_std]]
    return None

def preprocess_min_max(input_format):
    if input_format=="ktdr":
        dr_min,dr_max=0.000,14.143
        kt_min,kt_max=-10.232, 5.853
        return [[kt_min,kt_max],[dr_min,dr_max]]

    elif input_format=="4vec":
        e_min,e_max=0.226, 2804.960
        px_min,px_max=-946.552, 986.375
        py_min,py_max=-971.391, 895.821
        pz_min,pz_max=-2699.067, 2506.768
        return [[e_min,e_max],[px_min,px_max],[py_min,py_max],[pz_min,pz_max]]

    elif input_format=="relvec":
        ptfrac_min,ptfrac_max=0.000, 1.000
        deta_min,deta_max=-0.880, 0.880
        dphi_min,dphi_max=-0.845, 0.845
        return [[ptfrac_min,ptfrac_max],[deta_min,deta_max],[dphi_min,dphi_max]]
    return None

def preprocess(X,input_format,method="log"):
    mask = X == -1

    if method=="standardize":
        mean_std=preprocess_mean_std(input_format)
        for ii in range(X.shape[-1]):
            X[...,ii]=(X[...,ii]-mean_std[ii][0])/mean_std[ii][1]
        X[mask]=-50 #shift padding

    elif method=="linear":
        min_max=preprocess_min_max(input_format)
        for ii in range(X.shape[-1]):
            X[...,ii]=(X[...,ii]-min_max[ii][0])/(min_max[ii][1]-min_max[ii][0])
        X[mask]=-1 #keep padding

    elif method=="logit":
        min_max=preprocess_min_max(input_format)
        for ii in range(X.shape[-1]):
            X[...,ii]=np.log((X[...,ii]-min_max[ii][0])/(min_max[ii][1]-X[...,ii])) #log((x−a)/(b−x)) , linearize then logit
        X[mask]=-1 #keep padding

    elif method=="log" and input_format=="relvec":
        X[...,0]=-np.log(X[...,0]) #-log(x)
        X[mask]=-1 #keep padding

    elif method=="log":
        X = np.sign(X) * np.log1p(np.abs(X)) #sgn(x)*log(|x|+1)
        X[mask]=-10 #shift padding

    elif method=="logstd" and input_format=="relvec":
        mean_std=preprocess_log_mean_std(input_format)
        #X[...,0]=-np.log(X[...,0]) #-log(x)
        X[...,0]=np.log1p(X[...,0])
        for ii in range(0,X.shape[-1]):
            X[...,ii]=(X[...,ii]-mean_std[ii][0])/mean_std[ii][1]
        X[mask]=-10 #shift padding

    elif method=="logstd":
        mean_std=preprocess_log_mean_std(input_format)
        X = np.sign(X) * np.log1p(np.abs(X)) #sgn(x)*log(|x|+1)
        for ii in range(X.shape[-1]):
            X[...,ii]=(X[...,ii]-mean_std[ii][0])/mean_std[ii][1]
        X[mask]=-10 #shift padding

    elif method=="shiftpad":
        X[mask]=-3e3 #shift padding

    else:
        ValueError(f"Unknown pre-processing method: {method}")

    return X

def undo_preprocess(X,input_format,method="log"):
    X_new=torch.zeros(X.shape)

    if method=="standardize":
        if input_format=="ktdr":
            mask = X[...,-1] < -49
        else:
            mask = X[...,0] < -49
        mean_std=preprocess_mean_std(input_format)
        for ii in range(X_new.shape[-1]):
            X_new[...,ii]=X[...,ii]*mean_std[ii][1]+mean_std[ii][0]
        X_new[mask] = -1 #restore padding

    elif method=="linear":
        if input_format=="ktdr":
            mask = X[...,-1] < 0
        else:
            mask = X[...,0] < 0
        min_max=preprocess_min_max(input_format)
        for ii in range(X_new.shape[-1]):
            X_new[...,ii]=(min_max[ii][1]-min_max[ii][0])*X[...,ii] + min_max[ii][0]
        X_new[mask] = -1 #restore padding

    elif method=="logit":
        if input_format=="ktdr":
            mask = X[...,-1] < 0
        else:
            mask = X[...,0] < 0
        min_max=preprocess_min_max(input_format)
        for ii in range(X_new.shape[-1]):
            X_new[...,ii]=(min_max[ii][1]-min_max[ii][0])*(1./(1.+np.exp(-X[...,ii]))) + min_max[ii][0] #a + (b−a)·sigmoid(x), logistic then linearize
        X_new[mask] = -1 #restore padding

    elif method=="log" and input_format=="relvec":
        mask = X[...,0] < 0
        X_new=X
        X_new[...,0]=torch.exp(-X[...,0])
        X_new[mask]=-1 #restore padding

    elif method=="log":
        if input_format=="ktdr":
            mask = X[...,-1] < 0
        else:
            mask = X[...,0] < 0
        X_new = torch.sign(X) * torch.expm1(torch.abs(X))
        X_new[mask] = -1 #restore padding

    elif method=="logstd" and input_format=="relvec":
        mean_std=preprocess_log_mean_std(input_format)
        mask = X[...,0] < -9

        for ii in range(X_new.shape[-1]):
            X_new[...,ii]=X[...,ii]*mean_std[ii][1]+mean_std[ii][0]
        #X_new[...,0]=torch.exp(-X_new[...,0])
        X_new[...,0]=torch.expm1(X_new[...,0])

        X_new[mask]=-1 #restore padding

    elif method=="logstd":
        mean_std=preprocess_log_mean_std(input_format)
        mask = X[...,0] < -9

        for ii in range(X_new.shape[-1]):
            X_new[...,ii]=X[...,ii]*mean_std[ii][1]+mean_std[ii][0]
        X_new = torch.sign(X_new) * torch.expm1(torch.abs(X_new))

        X_new[mask] = -1 #restore padding

    elif method=="shiftpad":
        return X

    return X_new

def flatten_weight(X):
    #Predfined value reweight
    weights = torch.tensor([
        1.27576728e-08,5.90976731e-07,1.63552892e-06,3.32993680e-06,5.90960666e-06,9.69828631e-06,1.52290448e-05,2.31878681e-05,3.44328903e-05,5.08854061e-05,7.21865300e-05,1.02375102e-04,1.42328494e-04,1.96386489e-04,2.76166805e-04,3.56506239e-04,4.95540139e-04,6.30119723e-04,8.00000000e-04,1.06951872e-03,1.46842878e-03,1.72117040e-03,2.38095238e-03,3.33333333e-03,3.90625000e-03,5.34759358e-03,7.24637681e-03,9.90099010e-03,1.33333333e-02,1.75438596e-02,1.58730159e-02,3.12500000e-02,4.54545455e-02,5.26315789e-02,4.76190476e-02,1.11111111e-01,3.33333333e-01,1.66666667e-01,2.00000000e-01,3.33333333e-01,5.00000000e-01,2.50000000e-01,0.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00
        ], dtype=X.dtype, device=X.device)
    bin_edges = torch.tensor([
        -1.00000000e+00,5.62644844e+01,1.13528969e+02,1.70793457e+02,2.28057938e+02,2.85322418e+02,3.42586914e+02,3.99851379e+02,4.57115875e+02,5.14380371e+02,5.71644836e+02,6.28909302e+02,6.86173828e+02,7.43438293e+02,8.00702759e+02,8.57967285e+02,9.15231750e+02,9.72496216e+02,1.02976074e+03,1.08702515e+03,1.14428967e+03,1.20155420e+03,1.25881860e+03,1.31608313e+03,1.37334766e+03,1.43061206e+03,1.48787659e+03,1.54514111e+03,1.60240552e+03,1.65967004e+03,1.71693457e+03,1.77419897e+03,1.83146350e+03,1.88872803e+03,1.94599243e+03,2.00325696e+03,2.06052148e+03,2.11778589e+03,2.17505029e+03,2.23231494e+03,2.28957935e+03,2.34684375e+03,2.40410840e+03,2.46137280e+03,2.51863721e+03,2.57590186e+03
        ], dtype=X.dtype, device=X.device)

    values = X[..., 0] #E for [Nbatch, Nconstituent]

    # Find which bin each value belongs to
    idx = torch.bucketize(values, bin_edges) - 1

    # Clamp values outside the range
    idx = idx.clamp(0, len(weights) - 1)

    # Lookup weights
    w = weights[idx]

    return w

# ---------------------------------------------------------------------
# Schedulers and optimizers
# ---------------------------------------------------------------------
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
        phase = (2.0 * np.pi * (epoch - start_epoch) / period_epochs).item()
        oscillation_factor = 1.0 + cos_damping_amplitude * np.cos(phase).item()
        return max(baseline * oscillation_factor, 1e-12)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def get_cosine_scheduler(optimizer, num_epochs, eta_min=1e-6):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(num_epochs), 1),
        eta_min=eta_min,
    )

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_epochs, min_lr, max_lr,):
    """
    Linear warm-up followed by cosine decay.
    """

    if num_epochs <= num_warmup_steps:
        raise ValueError("num_epochs must be greater than num_warmup_steps")

    if min_lr < 0:
        raise ValueError("min_lr must be >= 0")

    if max_lr <= 0:
        raise ValueError("max_lr must be > 0")

    if min_lr > max_lr:
        raise ValueError("min_lr must be <= max_lr")

    # LambdaLR multiplies the optimizer's initial LR, so set the optimizer LR to max_lr.
    #if any(param_group["lr"] != max_lr for param_group in optimizer.param_groups): #FIXME, doesn't work when taking out block in adamW
    #    raise ValueError("optimizer learning rate must be set to max_lr")

    def lr_lambda(current_step):

        # Linear warm-up from min_lr -> max_lr
        if current_step < num_warmup_steps:
            lin_ratio = current_step / num_warmup_steps # 0 -> 1
            return (min_lr + (max_lr - min_lr) * lin_ratio) / max_lr # Convert absolute LR range to a multiplier.

        # Cosine decay from max_lr -> min_lr
        else:
            progress = ( current_step - num_warmup_steps) / max( 1, num_epochs - num_warmup_steps,)
            progress = min(progress, 1.0)
            cosine = 0.5 * ( 1.0 + np.cos(np.pi * progress)).item() # 0 -> 1
            return (min_lr + (max_lr - min_lr) * cosine) / max_lr # Convert absolute LR range to a multiplier.

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        return get_cosine_scheduler(
            optimizer=optimizer,
            num_epochs=args.epochs,
            eta_min=args.scheduler_min_lr,
        )
    if scheduler_name == "warmup_cosine":
        return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(int(0.2*args.epochs)-1, 0),
                num_epochs=args.epochs,
                min_lr=args.scheduler_min_lr,
                max_lr=args.lr,
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
        if args.architecture=="VQGPT" or not args.multi_loss:
            return torch.optim.AdamW(model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else: #Gradient descent on a plain categorical converges slowly and can be biased to zero in AdamW weight decay. Turn off weight-decay and give reasonable LR
            return torch.optim.AdamW([
                    {"params": [p for n, p in model.named_parameters() if n != "count_logits"], "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": [p for n, p in model.named_parameters() if n == "count_logits"], "lr": 1e-2, "weight_decay": 0.0},
            ])
    else:
        return torch.optim.Adam([
                {"params": [p for n, p in model.named_parameters() if n != "count_logits"], "lr": args.lr, "weight_decay": args.weight_decay},
                {"params": [model.count_logits], "lr": 1e-2, "weight_decay": 0.0},
        ])

# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------
def save_checkpoint(
    args,
    model,
    epoch,
    is_best,
    optimizer=None,
    scheduler=None,
    training_info=None,
    ckpt_name=None,
):

    #From args figure out how much to save
    save_mode = config_get(args, "save_mode", "full")
    if save_mode == "none":
        return None
    if save_mode not in ("full", "model"):
        raise ValueError("--save-mode must be one of: optimizer, model, none")

    #Get the path
    ckpt_dir = os.path.join(config_get(args, "log_dir"), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if ckpt_name is None:
        ckpt_name = f"epoch_{epoch:03d}.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    #Save the model info
    args_dict = dict(args) if isinstance(args, dict) else vars(args).copy()
    checkpoint_info = {
        "save_mode": save_mode,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "architecture": config_get(args, "architecture"),
        "model_config": getattr(model, "config", None),
    }

    #Add in the extra optimizer and learning info
    if save_mode == "full":
        checkpoint_info.update(
            {
                "args": args_dict,
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            }
        )
        checkpoint_info.update(training_info)

    #Actually save it
    torch.save(checkpoint_info, ckpt_path)
    save_torchpickle(checkpoint_info, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}", flush=True)
    if is_best:  # if best save to it's own file
        best_path=os.path.dirname(ckpt_path) +"/best.pt"
        save_torchpickle(checkpoint_info, best_path)
        print(f"Saved new best checkpoint to {best_path}", flush=True)
    return ckpt_path

def load_checkpoint(args):
    if len(args.model_checkpoint) == 0:
      raise ValueError("load_checkpoint requires --model-checkpoint")
    load_path = args.model_checkpoint[0] if isinstance(args.model_checkpoint, list) else args.model_checkpoint
    checkpoint_info = load_torchpickle(load_path)

    return checkpoint_info

def load_checkpoint_args(args, argvs, checkpoint_info=None):
    #Ignore things which are over-written by command-line
    ignore_args=[]
    for argv in argvs:
        if "--" in argv:
            ignore_args.append(argv.replace("--","").replace("-","_"))

    #Load checkpoint (or use the one passed in)
    checkpoint_info = load_checkpoint(args) if checkpoint_info is None else checkpoint_info

    #Load all the arguments
    checkpoint_info_args = checkpoint_info.get("args", {})
    for key in vars(args):
      if key in checkpoint_info_args and not key in ignore_args:
        setattr(args, key, checkpoint_info_args[key])

    return checkpoint_info

def load_checkpoint_model(args, checkpoint_info=None):
    if checkpoint_info is None:
        checkpoint_info = load_checkpoint(args)

    architecture = checkpoint_info.get("architecture") or config_get(args, "architecture")
    model = models_generative.build_model(architecture, checkpoint_info["model_config"])
    model.load_state_dict(checkpoint_info["model_state_dict"])

    return model

def load_checkpoint_scheduler_optimizer(args, checkpoint_info, optimizer, scheduler):
    if checkpoint_info is None:
        checkpoint_info = load_checkpoint(args)

    if checkpoint_info.get("optimizer_state_dict", None) is not None:
        optimizer.load_state_dict(checkpoint_info["optimizer_state_dict"])
    if scheduler is not None and checkpoint_info.get("scheduler_state_dict", None) is not None:
        scheduler.load_state_dict(checkpoint_info["scheduler_state_dict"])
    start_epoch = int(checkpoint_info.get("epoch", -1)) + 1
    print(f"Resuming after epoch {start_epoch}", flush=True)
    return start_epoch

def load_checkpoint_losses_lr(args, checkpoint_info, training_info):
    if checkpoint_info is None:
        checkpoint_info = load_checkpoint(args)

    training_info["best_loss"] = checkpoint_info.get("best_loss", None)
    if training_info["best_loss"] is None:
        training_info["best_loss"] = float("inf")
    training_info["best_epoch"] = checkpoint_info.get("best_epoch", None)
    training_info["patience_counter"] = checkpoint_info.get("patience_counter", 0)
    training_info["test_losses"] = list(checkpoint_info.get("test_losses", []))
    training_info["train_losses"] = list(checkpoint_info.get("train_losses", []))
    training_info["lr_history"] = list(checkpoint_info.get("lr_history", []))

def save_torchpickle(model, model_path):
    torch.save(model, model_path)
    return

def load_torchpickle(model_path, map_location="cpu"):
    #Note that torch load just unpickles a dictionary, which can be our checkpoint. The "model_state_dict" key holds the weights
    try:
        return torch.load(model_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=map_location)

# ---------------------------------------------------------------------
# Arguments and metadata saving
# ---------------------------------------------------------------------
def append_result_metadata(args, results):
    txt_path = os.path.join(args.log_dir, "arguments.txt")
    with open(txt_path, "a") as f:
        f.write("\n")
        for k, v in results.items():
            f.write(f"{k:20s} {v}\n")

def save_argument_metadata(args):
    #enumerate up the output directory if it exists, only do if not continueing
    if not getattr(args, "model_checkpoint", None):
        log_dir = args.log_dir
        i = 0
        while os.path.isdir(log_dir):
            i += 1
            log_dir = args.log_dir + f"_{i}"
        if args.plot_dir==args.log_dir: args.plot_dir=log_dir
        args.log_dir = log_dir

    #Loop over arguments and save them
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, "arguments.txt"), "w") as f:
        arg_dict = vars(args)
        for k, v in arg_dict.items():
            f.write(f"{k:20s} {v}\n")
    return

def common_training_args(parser):

    # data / io
    parser.add_argument("--input-format", type=str, choices=["ktdr","4vec","relvec","lundtree"], default="relvec", help="What format of inputs we are using")
    parser.add_argument("--num-constituents", type=int, default=20, help="Number of constituents")
    parser.add_argument("--preprocess", type=str, default=None, choices=["log","linear","standardize","logstd","shiftpad"], help="Preprocess the output (default: None")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",help="")

    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=1, help="DataLoader workers")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle training loader (default: True)")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable shuffle")
    parser.add_argument("--seed", type=int, default=0,help="")
    parser.add_argument("--block-size", type=int, default=None, help="Data streaming: jets per contiguous read from the .h5 (default: sized for a ~160 MB shuffle buffer)")
    parser.add_argument("--blocks-per-buffer", type=int, default=32, help="Data streaming: random blocks mixed per shuffle buffer (RAM per worker ~ 2-3 x block-size x this x jet size)")
    parser.add_argument("--max-jets", type=int, default=None, help="Use at most this many jets per file (quick tests)")

     # logging / checkpointing
    parser.add_argument("--log-dir", dest="log_dir", type=str, default="models/test",help="Logging directory")
    parser.add_argument("--plot-dir", dest="plot_dir", type=str, default=None, help="Output directory for plots. Default: use --log-dir / inferred checkpoint log directory",)
    parser.add_argument("--save-mode", type=str, default="full", choices=["full", "model", "none"], help="Saved training artifact: full info, model-state only, or none")
    parser.add_argument("--model-checkpoint", "--checkpoint", type=str, nargs="+",default=[],help="Path(s) to model/checkpoint to load")

    # training
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Optimizer weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping. Set <=0 to disable")
    parser.add_argument("--scheduler", type=str, default="warmup_cosine", choices=["none", "cos_damping", "cosine", "warmup_cosine", "plateau"], help="Learning rate scheduler")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6, help="Minimum LR for cosine/plateau schedulers")
    parser.add_argument("--plateau-factor", type=float, default=0.5, help="LR multiplier for --scheduler plateau")
    parser.add_argument("--plateau-patience", type=int, default=2, help="Plateau epochs before reducing LR")
    parser.add_argument("--cos-damping-start-epoch", type=int, default=0, help="Epoch where cosine damping starts")
    parser.add_argument("--cos-damping-end-epoch", type=int, default=None, help="Epoch where cosine damping ends")
    parser.add_argument("--cos-damping-final-lr", type=float, default=5e-5, help="Final LR for cosine damping")
    parser.add_argument("--cos-damping-amplitude", type=float, default=0.0, help="Cosine oscillation amplitude")
    parser.add_argument("--cos-damping-period-epochs", type=float, default=1.0, help="Cosine oscillation period in epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # misc
    parser.add_argument("--mask-bad", action="store_true", default=False, help="Mask bad generated consitiuents")
    parser.add_argument("--test", action="store_true", default=False, help="Setup with reduced training size for easy debugging/testing")

    # plotting options
    parser.add_argument("--validation-size",type=int, default=10000, help="How many images to generate and make plots for")
    parser.add_argument("--validation-samples", type=str, default=None, help="NPZ file to write/use that stores the numpy arrays for the validation")
    parser.add_argument("--hist2d-xrange", type=float, nargs=2, default=[-1,10], help="2D Lund histogram x range: xmin xmax",)
    parser.add_argument("--hist2d-yrange", type=float, nargs=2, default=[-5,7], help="2D Lund histogram y range: ymin ymax",)
    parser.add_argument("--hist2d-bins", type=int, nargs=2, default=[20, 20], help="2D Lund histogram bins: xbins ybins",)
    parser.add_argument("--hist2d-shape", "--hist2d_shape", "--hist2d-layout", dest="hist2d_shape", type=int, nargs=2, default=None, metavar=("ROWS", "COLS"), help="Manual 2D Lund subplot shape. Default auto: 2 -> 1x2, 3 -> 1x3, 4 -> 2x2",)
    parser.add_argument("--hist1d-ranges", type=float, nargs="+", default=None, help="Flattened 1D ranges: kt_min kt_max dr_min dr_max",)
    parser.add_argument("--hist1d-bins", type=int, default=30, help="Number of bins for 1D histograms",)
    parser.add_argument("--hist-ratio-min-count", type=int, default=5, help="Mask 2D relative-difference bins with fewer original entries than this",)
    parser.add_argument("--hist-ratio-vmax", type=float, default=1.0, help="Symmetric color limit for 2D fractional relative-difference plots",)
