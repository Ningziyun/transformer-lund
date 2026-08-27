import numpy as np
import math

import torch

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
    return None

def preprocess(X,input_format,method="log"):
    mask = X == -1

    if method=="standardize":
        mean_std=preprocess_mean_std(input_format)
        for ii in range(X.shape[-1]):
            X[:,ii]=(X[:,ii]-mean_std[ii][0])/mean_std[ii][1]
        X[mask]=-50 #shift padding

    elif method=="linear":
        min_max=preprocess_min_max(input_format)
        for ii in range(X.shape[-1]):
            X[:,ii]=(X[:,ii]-min_max[ii][0])/(min_max[ii][1]-min_max[ii][0])
        X[mask]=-1 #shift padding

    elif method=="log":
        X[...] = torch.sign(X) * torch.log1p(torch.abs(X)) #sgn(x)*log(|x|+1)
        X[mask]=-10 #restore padding

    elif method=="shiftnan":
        X[mask]=-3e3 #shift padding

    else:
        ValueError(f"Unknown pre-processing method: {method}")

def undo_preprocess(X,input_format,method="log"):
    X_new=torch.zeros(X.shape)

    if method=="standardize":
        if input_format=="4vec":
            mask = X[:,:,0] < -49
        else:
            mask = X[:,:,-1] < -49
        mean_std=preprocess_mean_std(input_format)
        for ii in range(X_new.shape[-1]):
            X_new[:,:,ii]=X[:,:,ii]*mean_std[ii][1]+mean_std[ii][0]
        X_new[mask] = -1

    elif method=="linear":
        if input_format=="4vec":
            mask = X[:,:,0] < 0
        else:
            mask = X[:,:,-1] < 0
        min_max=preprocess_min_max(input_format)
        for ii in range(X_new.shape[-1]):
            X_new[:,:,ii]=(min_max[ii][1]-min_max[ii][0])*X[:,:,ii] + min_max[ii][0]
        X_new[mask] = -1

    elif method=="log":
        if input_format=="4vec":
            mask = X[:,:,0] < 0
        else:
            mask = X[:,:,-1] < 0
        X_new = torch.sign(X) * torch.expm1(torch.abs(X))
        X_new[mask] = -1

    elif method=="shiftnan":
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

def format_input(X, args, device):
    if args.mixed_loss:
        X, mask=X
    else:
        mask=None
    X = X.to(device)
    if mask is not None: mask=mask.to(device)
    return X, mask
