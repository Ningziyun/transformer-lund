#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 22:24:30 2025

@author: ningyan
"""

import uproot
import awkward as ak
import numpy as np

# Read the root files
with uproot.open("originalJets_qcd.root") as f:
    tree = f["tree"]
    constit_pt = tree["constit_pt"].array(library="ak")
    constit_eta = tree["constit_eta"].array(library="ak")
    constit_phi = tree["constit_phi"].array(library="ak")

# Construct the constit_e（all 0）
constit_e = ak.Array([[0.0] * len(entry) for entry in constit_pt])

# Make sure everything is in float32（requirement ofUproot）
constit_pt = ak.values_astype(constit_pt, np.float32)
constit_eta = ak.values_astype(constit_eta, np.float32)
constit_phi = ak.values_astype(constit_phi, np.float32)
constit_e = ak.values_astype(constit_e, np.float32)

# Construct record type Awkward Array
data = {
    "constit_pt": constit_pt,
    "constit_eta": constit_eta,
    "constit_phi": constit_phi,
    "constit_e": constit_e,
}

# Save the file
with uproot.recreate("originalJets_qcd.root") as fout:
    fout.mktree("tree", {
        "constit_pt": "var * float32",
        "constit_eta": "var * float32",
        "constit_phi": "var * float32",
        "constit_e": "var * float32",
    })
    fout["tree"].extend(data)
