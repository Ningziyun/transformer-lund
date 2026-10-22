import os,sys
import numpy as np
import math
import h5py

import torch
from torch.utils.data import DataLoader

import fastjet
import ljpHelpers

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
class ktdr_dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, NConstituents=20, add_mask=False, add_conditional=False, preprocess=None):
        super(ktdr_dataset, self).__init__()
        self.n_features=2
        self.pad_value=-1

        self.add_mask=add_mask
        self.preprocess=preprocess
        self.add_conditional=add_conditional

        #Claude says its faster to stack and index, since slicing the h5py still loads everything in memory anyways
        with h5py.File(file_path, "r") as f:
            const = f["lundplane"]
            self.X = np.stack([np.asarray(const[k][:, :NConstituents], dtype=np.float32) for k in ["kt","dr"]], axis=-1)

            if self.add_conditional:
                j = f["jet"]
                jet = np.stack([np.asarray(j[k][:], dtype=np.float64) for k in ["E","PX","PY","PZ"]], axis=-1)

        #Mask which stores if padded. True/False=Padded/unpadded
        if add_mask or self.add_conditional:
            self.mask=np.all(self.X == np.float32(self.pad_value), axis=-1)
            self.n_real = (~self.mask).sum(axis=1).reshape(-1,1)

        #Make the conditioning info once we know masks
        if self.add_conditional:
            pt,eta,phi,m,_=jet_PtEtaPhiM(jet)
            self.jet_data = np.stack([pt, eta, phi, m, self.n_real], axis=-1).astype(np.float32).reshape(self.mask.shape[0],-1)

    def __getitem__(self, index):
        X = torch.from_numpy(self.X[index])
        if self.add_mask and self.add_conditional:
            return [X, torch.from_numpy(self.jet_data[index]), torch.from_numpy(self.mask[index])]
        elif self.add_mask:
            return [X, torch.from_numpy(self.mask[index])]
        elif self.add_conditional:
            return [X, torch.from_numpy(self.jet_data[index])]
        return X

    def __len__(self):
        return len(self.X)

class constit_dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, NConstituents=20, add_mask=False, add_conditional=False, preprocess=None):
        super(constit_dataset, self).__init__()
        self.n_features=4
        self.pad_value=-10

        self.add_mask=add_mask
        self.preprocess=preprocess
        self.add_conditional=add_conditional

        #Claude says its faster to stack and index, since slicing the h5py still loads everything in memory anyways
        with h5py.File(file_path, "r") as f:
            const = f["constituents"]
            self.X = np.stack([np.asarray(const[k][:, :NConstituents], dtype=np.float32) for k in ["E","PX","PY","PZ"]], axis=-1)

            if self.add_conditional:
                j = f["jet"]
                jet = np.stack([np.asarray(j[k][:], dtype=np.float64) for k in ["E","PX","PY","PZ"]], axis=-1)

        #Mask which stores if padded. True/False=Padded/unpadded
        if add_mask or self.add_conditional:
            self.mask=np.all(self.X == np.float32(self.pad_value), axis=-1)
            self.n_real = (~self.mask).sum(axis=1).reshape(-1,1)

        #Make the conditioning info once we know masks
        if self.add_conditional:
            pt,eta,phi,m,_=jet_PtEtaPhiM(jet)
            self.jet_data = np.stack([pt, eta, phi, m, self.n_real], axis=-1).astype(np.float32).reshape(self.mask.shape[0],-1)

    def __getitem__(self, index):
        X = torch.from_numpy(self.X[index])
        if self.add_mask and self.add_conditional:
            return [X, torch.from_numpy(self.jet_data[index]), torch.from_numpy(self.mask[index])]
        elif self.add_mask:
            return [X, torch.from_numpy(self.mask[index])]
        elif self.add_conditional:
            return [X, torch.from_numpy(self.jet_data[index])]
        return X

    def __len__(self):
        return len(self.X)

class constit_relative_dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, NConstituents=20, add_mask=False, add_conditional=False, preprocess=None):
        super(constit_relative_dataset, self).__init__()
        self.n_features=3
        self.pad_value=-10

        self.add_mask=add_mask
        self.preprocess=preprocess
        self.add_conditional=add_conditional

        #Claude says its faster to stack and index, since slicing the h5py still loads everything in memory anyways
        with h5py.File(file_path, "r") as f:
            const = f["relative_constituents"]
            self.X = np.stack([np.asarray(const[k][:, :NConstituents], dtype=np.float32) for k in ["ptfrac","deta","dphi"]], axis=-1)

            if self.add_conditional:
                j = f["jet"]
                jet = np.stack([np.asarray(j[k][:], dtype=np.float64) for k in ["E","PX","PY","PZ"]], axis=-1)

        #Mask which stores if padded. True/False=Padded/unpadded
        if add_mask or self.add_conditional:
            self.mask=np.all(self.X == np.float32(self.pad_value), axis=-1)
            self.n_real = (~self.mask).sum(axis=1).reshape(-1,1)

        #Make the conditioning info once we know masks
        if self.add_conditional:
            pt,eta,phi,m,_=jet_PtEtaPhiM(jet)
            self.jet_data = np.stack([pt, eta, phi, m, self.n_real], axis=-1).astype(np.float32).reshape(self.mask.shape[0],-1)
            #self.jet_data = np.concatenate([jet, self.n_real], axis=-1).astype(np.float32)

    def __getitem__(self, index):
        X = torch.from_numpy(self.X[index])
        if self.add_mask and self.add_conditional:
            return [X, torch.from_numpy(self.jet_data[index]), torch.from_numpy(self.mask[index])]
        elif self.add_mask:
            return [X, torch.from_numpy(self.mask[index])]
        elif self.add_conditional:
            return [X, torch.from_numpy(self.jet_data[index])]
        return X

    def __len__(self):
        return len(self.X)


def get_loader(filename, args):
    '''
    Handles all the data-loader configurations
    '''

    #Interface for differnt argparser setups between vqvae and generative training
    add_mask=config_get(args, "multi_loss", False)
    add_conditional=config_get(args, "conditional", False)
    if config_get(args, "codebook_size"):
        add_mask=True
        add_conditional=True

    input_format=config_get(args, "input_format","")
    num_constituents=config_get(args, "num_constituents", 0)
    preprocess=config_get(args, "preprocess", False)
    batch_size=config_get(args, "batch_size", 0)
    num_workers=config_get(args, "num_workers", 0)
    shuffle=config_get(args, "shuffle", True)

    #Actually load info
    if input_format=="ktdr":
        dataset = ktdr_dataset(filename, NConstituents=num_constituents, add_mask=add_mask, add_conditional=add_conditional, preprocess=preprocess)
    elif input_format=="4vec":
        dataset = constit_dataset(filename, NConstituents=num_constituents, add_mask=add_mask, add_conditional=add_conditional, preprocess=preprocess)
    elif input_format=="relvec":
        add_conditional=True
        dataset = constit_relative_dataset(filename, NConstituents=num_constituents, add_mask=add_mask, add_conditional=add_conditional, preprocess=preprocess)
    else:
       ValueError(f"No known input format {input_format}") 

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return loader

def get_loaders(args):
    '''
    Calls above
    '''

    train_loader=get_loader(args.train_file, args)
    test_loader=get_loader(args.test_file, args)

    return train_loader, test_loader

# ---------------------------------------------------------------------
# Data format handling
# ---------------------------------------------------------------------
def format_input_args(args):

    func_args={"unpack_mask":False, "unpack_jet": False, "device": args.device}

    input_format=config_get(args,"input_format","")
    multi_loss=config_get(args,"multi_loss",False)
    conditional=config_get(args,"conditional",False)

    if (input_format=="relvec" and multi_loss) or (conditional and multi_loss):
        func_args["unpack_mask"]=True
        func_args["unpack_jet"]=True
    elif input_format=="relvec" or conditional:
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

def load_checkpoint_args(args, argvs, checkpoint_model=None):
    #Ignore things which are over-written by command-line
    ignore_args=[]
    for argv in argvs:
        if "--" in argv:
            ignore_args.append(argv.replace("--","").replace("-","_"))

    #Load checkpoint
    if checkpoint_model is None:
        checkpoint_info = load_checkpoint(args)

    #Load all the arguments
    checkpoint_info_args = checkpoint_info.get("args", {})
    for key in vars(args):
      if key in checkpoint_info_args and not key in ignore_args:
        setattr(args, key, checkpoint_info_args[key])

    return checkpoint_info

def load_checkpoint_model(args, shape, cond_shape=[0], checkpoint_info=None):
    if checkpoint_info is None:
        checkpoint_info = load_checkpoint(args)

    model = build_unbinned_model(args, shape, cond_shape)
    model.load_state_dict(checkpoint_info["model_state_dict"])

    return model

def load_checkpoint_scheduler_optimizer(args, checkpoint_info, optimizer, scheduler):
    if checkpoint_info.get("optimizer_state_dict", None) is not None:
        optimizer.load_state_dict(checkpoint_info["optimizer_state_dict"])
    if scheduler is not None and checkpoint_info.get("scheduler_state_dict", None) is not None:
        scheduler.load_state_dict(checkpoint_info["scheduler_state_dict"])
    start_epoch = int(checkpoint_info.get("epoch", -1)) + 1
    print(f"Resuming after epoch {start_epoch}", flush=True)
    return start_epoch

def load_checkpoint_losses_lr(args, checkpoint_info, training_info):
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
    parser.add_argument("--input_format", type=str, choices=["ktdr","4vec","relvec"], default="relvec", help="What format of inputs we are using")
    parser.add_argument("--num-constituents", type=int, default=20, help="Number of constituents")
    parser.add_argument("--preprocess", type=str, default=None, choices=["log","linear","standardize","logstd","shiftpad"], help="Preprocess the output (default: None")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",help="")

    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=1, help="DataLoader workers")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle training loader (default: True)")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable shuffle")
    parser.add_argument("--seed", type=int, default=0,help="")

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
    parser.add_argument("--hist2d-xrange", type=float, nargs=2, default=[-1,10], help="2D Lund histogram x range: xmin xmax",)
    parser.add_argument("--hist2d-yrange", type=float, nargs=2, default=[-5,7], help="2D Lund histogram y range: ymin ymax",)
    parser.add_argument("--hist2d-bins", type=int, nargs=2, default=[20, 20], help="2D Lund histogram bins: xbins ybins",)
    parser.add_argument("--hist2d-shape", "--hist2d_shape", "--hist2d-layout", dest="hist2d_shape", type=int, nargs=2, default=None, metavar=("ROWS", "COLS"), help="Manual 2D Lund subplot shape. Default auto: 2 -> 1x2, 3 -> 1x3, 4 -> 2x2",)
    parser.add_argument("--hist1d-ranges", type=float, nargs="+", default=None, help="Flattened 1D ranges: kt_min kt_max dr_min dr_max",)
    parser.add_argument("--hist1d-bins", type=int, default=30, help="Number of bins for 1D histograms",)
    parser.add_argument("--hist-ratio-min-count", type=int, default=5, help="Mask 2D relative-difference bins with fewer original entries than this",)
    parser.add_argument("--hist-ratio-vmax", type=float, default=1.0, help="Symmetric color limit for 2D fractional relative-difference plots",)
