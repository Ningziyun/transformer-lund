import numpy as np
import math

import torch

import fastjet
import ljpHelpers

# ---------------------------------------------------------------------
# Make the lundplane
# ---------------------------------------------------------------------
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
