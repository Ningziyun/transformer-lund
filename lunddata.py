#!/usr/bin/env python3

import argparse
import os, sys, glob
import ROOT
import uproot
import awkward as ak
import numpy as np
import random
import h5py

from helpers import make_lundplane

def load_and_lundplane(files, treename, outdir="inputFiles/", outname="qcd.h5", train_test_split=0.8, seed=0, file_format="topbenchmark"):

    #make the output file
    if not os.path.exists(outdir):
      os.makedirs(outdir)
    outpath = os.path.join(outdir, outname)
    outfile_train=h5py.File(outpath.replace(".h5","_train.h5"), "w")
    outfile_test=h5py.File(outpath.replace(".h5","_test.h5"), "w")

    #Set the padding size
    Npad_const=100
    Npad_ljp=20
    Ncount=0

    random.seed(seed)

    #Loop over input files
    for nfile,filename in enumerate(files):

      #Get the tree via either old root or uproot
      if file_format=="topbenchmark":
        file = ROOT.TFile(filename)
        tree = file.Get(treename)
      elif file_format=="jetclass":
        file = uproot.open(filename)
        tree = file[treename]

      if not tree:
        continue
      print(f"Running file {filename} and tree {treename}")
       
      if file_format=="topbenchmark":

        #If first file make the hdf5 format, fixing the size to tree length since will be 1 file
        if nfile==0:
          grp_const_train = outfile_train.create_group("constituents")
          dset_E_train = grp_const_train.create_dataset("E", shape=(0, Npad_const),maxshape=(None, Npad_const), dtype="float32")
          dset_PX_train = grp_const_train.create_dataset("PX", shape=(0, Npad_const),maxshape=(None, Npad_const), dtype="float32")
          dset_PY_train = grp_const_train.create_dataset("PY", shape=(0, Npad_const),maxshape=(None, Npad_const), dtype="float32")
          dset_PZ_train = grp_const_train.create_dataset("PZ", shape=(0, Npad_const),maxshape=(None, Npad_const), dtype="float32")

          grp_ljp_train = outfile_train.create_group("lundplane")
          dset_kt_train = grp_ljp_train.create_dataset("kt", shape=(0, Npad_ljp),maxshape=(None, Npad_ljp), dtype="float32")
          dset_dr_train = grp_ljp_train.create_dataset("dr", shape=(0, Npad_ljp),maxshape=(None, Npad_ljp), dtype="float32")

          grp_const_test = outfile_test.create_group("constituents")
          dset_E_test = grp_const_test.create_dataset("E", shape=(0, Npad_const),maxshape=(None, Npad_const), dtype="float32")
          dset_PX_test = grp_const_test.create_dataset("PX", shape=(0, Npad_const),maxshape=(None, Npad_const), dtype="float32")
          dset_PY_test = grp_const_test.create_dataset("PY", shape=(0, Npad_const),maxshape=(None, Npad_const), dtype="float32")
          dset_PZ_test = grp_const_test.create_dataset("PZ", shape=(0, Npad_const),maxshape=(None, Npad_const), dtype="float32")

          grp_ljp_test = outfile_test.create_group("lundplane")
          dset_kt_test = grp_ljp_test.create_dataset("kt", shape=(0, Npad_ljp),maxshape=(None, Npad_ljp), dtype="float32")
          dset_dr_test = grp_ljp_test.create_dataset("dr", shape=(0, Npad_ljp),maxshape=(None, Npad_ljp), dtype="float32")

        # Loop through all events one at at time and store jet/const info
        for index, event in enumerate(tree):
          if index%1000 == 0: print(f"index={index},Njets={Ncount}")

          #make an array size [1, Nconst, 4-vector]
          constituents=np.zeros([1,Npad_const,4])
          
          #Get the info from root branch and save into numpy array
          Nconst=len(event.constit_pt)
          constit_pt = np.asarray(event.constit_pt)
          constit_eta = np.asarray(event.constit_eta)
          constit_phi = np.asarray(event.constit_phi)
          px = constit_pt * np.cos(constit_phi)
          py = constit_pt * np.sin(constit_phi)
          pz = constit_pt * np.sinh(constit_eta)
          E=np.sqrt(px**2+py**2+pz**2)
          for jj in range(min(Nconst,Npad_const)):
            constituents[0,jj,:]=[E[jj],px[jj],py[jj],pz[jj]]

          #Make the lund-jet place
          ljp=make_lundplane(constituents, Npad_ljp)
          constituents[constituents==0]=-1 #set pad to -1

          #train/test split
          if random.random()<train_test_split:
              dset_E=dset_E_train
              dset_PX=dset_PX_train
              dset_PY=dset_PY_train
              dset_PZ=dset_PZ_train
              dset_kt=dset_kt_train
              dset_dr=dset_dr_train
          else:
              dset_E=dset_E_test
              dset_PX=dset_PX_test
              dset_PY=dset_PY_test
              dset_PZ=dset_PZ_test
              dset_kt=dset_kt_test
              dset_dr=dset_dr_test

          #Reshape the input with new inputs
          dset_E.resize(dset_E.shape[0]+constituents.shape[0], axis=0)
          dset_PX.resize(dset_PX.shape[0]+constituents.shape[0], axis=0)
          dset_PY.resize(dset_PY.shape[0]+constituents.shape[0], axis=0)
          dset_PZ.resize(dset_PZ.shape[0]+constituents.shape[0], axis=0)
          dset_kt.resize(dset_kt.shape[0]+constituents.shape[0], axis=0)
          dset_dr.resize(dset_dr.shape[0]+constituents.shape[0], axis=0)

          #Add the new inputs
          dset_E[-constituents.shape[0]:,:] = constituents[:,:,0]
          dset_PX[-constituents.shape[0]:,:] = constituents[:,:,1]
          dset_PY[-constituents.shape[0]:,:] = constituents[:,:,2]
          dset_PZ[-constituents.shape[0]:,:] = constituents[:,:,3]
          dset_kt[-constituents.shape[0]:,:] = ljp[:,:,0]
          dset_dr[-constituents.shape[0]:,:] = ljp[:,:,1]
          Ncount+=1

      if file_format=="jetclass":

        #If first file make the hdf5 format, will leave it empty in length and fill it up as we go on-the-fly
        if nfile==0:
          grp_jet_train = outfile_train.create_group("jet")
          dset_jetE_train = grp_jet_train.create_dataset("E", shape=(0,), maxshape=(None,), dtype="float32")
          dset_jetPX_train = grp_jet_train.create_dataset("PX", shape=(0,), maxshape=(None,), dtype="float32")
          dset_jetPY_train = grp_jet_train.create_dataset("PY", shape=(0,), maxshape=(None,), dtype="float32")
          dset_jetPZ_train = grp_jet_train.create_dataset("PZ", shape=(0,), maxshape=(None,), dtype="float32")

          grp_const_train = outfile_train.create_group("constituents")
          dset_E_train = grp_const_train.create_dataset("E", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PX_train = grp_const_train.create_dataset("PX", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PY_train = grp_const_train.create_dataset("PY", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PZ_train = grp_const_train.create_dataset("PZ", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")

          grp_relconst_train = outfile_train.create_group("relative_constituents")
          dset_ptfrac_train = grp_relconst_train.create_dataset("ptfrac", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_deta_train = grp_relconst_train.create_dataset("deta", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_dphi_train = grp_relconst_train.create_dataset("dphi", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_m_train = grp_relconst_train.create_dataset("m", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")

          grp_ljp_train = outfile_train.create_group("lundplane")
          dset_kt_train = grp_ljp_train.create_dataset("kt", shape=(0,Npad_ljp), maxshape=(None, Npad_const), dtype="float32")
          dset_dr_train = grp_ljp_train.create_dataset("dr", shape=(0,Npad_ljp), maxshape=(None, Npad_const), dtype="float32")

          grp_jet_test = outfile_test.create_group("jet")
          dset_jetE_test = grp_jet_test.create_dataset("E", shape=(0,), maxshape=(None,), dtype="float32")
          dset_jetPX_test = grp_jet_test.create_dataset("PX", shape=(0,), maxshape=(None,), dtype="float32")
          dset_jetPY_test = grp_jet_test.create_dataset("PY", shape=(0,), maxshape=(None,), dtype="float32")
          dset_jetPZ_test = grp_jet_test.create_dataset("PZ", shape=(0,), maxshape=(None,), dtype="float32")

          grp_const_test = outfile_test.create_group("constituents")
          dset_E_test = grp_const_test.create_dataset("E", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PX_test = grp_const_test.create_dataset("PX", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PY_test = grp_const_test.create_dataset("PY", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PZ_test = grp_const_test.create_dataset("PZ", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")

          grp_relconst_test = outfile_test.create_group("relative_constituents")
          dset_ptfrac_test = grp_relconst_test.create_dataset("ptfrac", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_deta_test = grp_relconst_test.create_dataset("deta", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_dphi_test = grp_relconst_test.create_dataset("dphi", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_m_test = grp_relconst_test.create_dataset("m", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")

          grp_ljp_test = outfile_test.create_group("lundplane")
          dset_kt_test = grp_ljp_test.create_dataset("kt", shape=(0,Npad_ljp), maxshape=(None, Npad_const), dtype="float32")
          dset_dr_test = grp_ljp_test.create_dataset("dr", shape=(0,Npad_ljp), maxshape=(None, Npad_const), dtype="float32")

        #Loop over the input files in chunks via uproot
        chunk_size = 2**11
        batch=0
        for index in range(0, tree.num_entries, chunk_size): #loading in chunks of rows at a time
          stop_index=min(index + chunk_size, tree.num_entries)
          print(f"batch={batch},index={index},Njets={Ncount}")

          #Assume input is dimensions [Nevents, Nconstituents, 4-vec] with 4-vec=[E,px,py,pz], and Nconst will be jagged dimension
          unpacked_const = tree.arrays(["part_energy","part_px", "part_py", "part_pz"], entry_start=index, entry_stop=stop_index, library="np")
          unpacked_jet = tree.arrays(["jet_pt","jet_eta", "jet_phi", "jet_energy"], entry_start=index, entry_stop=stop_index, library="np")
          unpacked_aux = tree.arrays(["part_deta","part_dphi"], entry_start=index, entry_stop=stop_index, library="np")
          Njet=len(unpacked_const["part_energy"])

          #Make the empty array we will fill
          constituents=np.zeros([Njet,Npad_const,4])
          
          #Now fill it with the 4-vectors
          for ii in range(Njet):
            for jj in range(min(len(unpacked_const["part_energy"][ii]),Npad_const)):
              constituents[ii,jj,:]=[unpacked_const["part_energy"][ii][jj],unpacked_const["part_px"][ii][jj],unpacked_const["part_py"][ii][jj],unpacked_const["part_pz"][ii][jj]]

          #Also load in the deta and dphi from JetClass, this seems to have buggy features with const delta_eta>>1
          part_deta=np.zeros([Njet,Npad_const])
          part_dphi=np.zeros([Njet,Npad_const])
          for ii in range(Njet):
            for jj in range(min(len(unpacked_aux["part_deta"][ii]),Npad_const)):
              part_deta[ii,jj]=unpacked_aux["part_deta"][ii][jj]
              part_dphi[ii,jj]=unpacked_aux["part_dphi"][ii][jj]
          part_dr=np.sqrt(part_deta**2+part_dphi**2)

          #Make the relative info
          E  = constituents[..., 0]
          px = constituents[..., 1]
          py = constituents[..., 2]
          pz = constituents[..., 3]

          # Real constituents
          const_mask = (E > 0) & (part_dr<0.8) # Existing padding convention plus removing weird wide-angles const
          sort_idx = np.argsort(~const_mask, axis=1, kind="stable") # Sort so real constituents (True) come first, padding (False) last
          eps = 1e-12

          # Full jet 4-vector, agree with the unpacked version from JetClass directly within a percent
          jet_E  = np.sum(np.where(const_mask, E, 0.0), axis=1)
          jet_px = np.sum(np.where(const_mask, px, 0.0), axis=1)
          jet_py = np.sum(np.where(const_mask, py, 0.0), axis=1)
          jet_pz = np.sum(np.where(const_mask, pz, 0.0), axis=1)

          jet_pt = np.hypot(jet_px, jet_py)
          jet_eta = np.arcsinh(jet_pz / np.maximum(jet_pt, eps))
          jet_phi = np.arctan2(jet_py, jet_px)
          jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2)
          jet_m = np.sqrt(np.maximum(jet_E**2 - jet_p**2, 0.0))

          #jet_pt=unpacked_jet["jet_pt"]
          #jet_eta_diff=unpacked_jet["jet_eta"]
          #jet_phi_diff=unpacked_jet["jet_phi"]
          #jet_E=unpacked_jet["jet_energy"]
          #jet_m = np.sqrt(np.maximum(jet_E**2 - jet_p**2, 0.0))

          # Constituent info
          const_pt = np.hypot(px, py)
          const_eta = np.arcsinh(pz / np.maximum(const_pt, eps))
          const_phi = np.arctan2(py, px)
          const_p = np.sqrt(px**2 + py**2 + pz**2)
          const_m = np.sqrt(np.maximum(E**2 - const_p**2, 0.0))

          # Relative coordinates
          const_ptfrac = const_pt/jet_pt[:, None]
          const_deta = const_eta - jet_eta[:, None]
          const_dphi = const_phi - jet_phi[:, None]
          const_dphi = np.arctan2(np.sin(const_dphi), np.cos(const_dphi)) # Wrap Delta phi into [-pi, pi]

          #Remove out the weird deta issues for sure via the mask and sort to end
          E[~const_mask]=-1
          px[~const_mask]=-1
          py[~const_mask]=-1
          pz[~const_mask]=-1
          const_ptfrac[~const_mask]=-1
          const_dphi[~const_mask]=-1
          const_deta[~const_mask]=-1
          const_m[~const_mask]=-1

          E          = np.take_along_axis(E,          sort_idx, axis=1)
          px         = np.take_along_axis(px,         sort_idx, axis=1)
          py         = np.take_along_axis(py,         sort_idx, axis=1)
          pz         = np.take_along_axis(pz,         sort_idx, axis=1)
          const_ptfrac   = np.take_along_axis(const_ptfrac,   sort_idx, axis=1)
          const_deta  = np.take_along_axis(const_deta,  sort_idx, axis=1)
          const_dphi  = np.take_along_axis(const_dphi,  sort_idx, axis=1)
          const_m  = np.take_along_axis(const_m,  sort_idx, axis=1)

          #Make the lund-jet place
          ljp=make_lundplane(constituents, Npad_ljp)
          constituents[constituents==0]=-1 #set pad to -1

          #train/test split
          if random.random()<train_test_split:
              dset_jetE=dset_jetE_train
              dset_jetPX=dset_jetPX_train
              dset_jetPY=dset_jetPY_train
              dset_jetPZ=dset_jetPZ_train

              dset_E=dset_E_train
              dset_PX=dset_PX_train
              dset_PY=dset_PY_train
              dset_PZ=dset_PZ_train

              dset_ptfrac=dset_ptfrac_train
              dset_deta=dset_deta_train
              dset_dphi=dset_dphi_train
              dset_m=dset_m_train

              dset_kt=dset_kt_train
              dset_dr=dset_dr_train
          else:
              dset_jetE=dset_jetE_test
              dset_jetPX=dset_jetPX_test
              dset_jetPY=dset_jetPY_test
              dset_jetPZ=dset_jetPZ_test

              dset_E=dset_E_test
              dset_PX=dset_PX_test
              dset_PY=dset_PY_test
              dset_PZ=dset_PZ_test

              dset_ptfrac=dset_ptfrac_test
              dset_deta=dset_deta_test
              dset_dphi=dset_dphi_test
              dset_m=dset_m_test

              dset_kt=dset_kt_test
              dset_dr=dset_dr_test

          #Reshape the input with new inputs
          old_size = dset_jetE.shape[0]
          new_size = old_size + jet_E.shape[0]

          dset_jetE.resize(new_size, axis=0)
          dset_jetPX.resize(new_size, axis=0)
          dset_jetPY.resize(new_size, axis=0)
          dset_jetPZ.resize(new_size, axis=0)

          dset_E.resize(new_size, axis=0) 
          dset_PX.resize(new_size, axis=0)
          dset_PY.resize(new_size, axis=0)
          dset_PZ.resize(new_size, axis=0)

          dset_ptfrac.resize(new_size, axis=0)
          dset_dphi.resize(new_size, axis=0)
          dset_deta.resize(new_size, axis=0)
          dset_m.resize(new_size, axis=0)

          dset_kt.resize(new_size, axis=0)
          dset_dr.resize(new_size, axis=0)

          #Add the new inputs
          dset_jetE[old_size:new_size] = jet_E
          dset_jetPX[old_size:new_size] = jet_px
          dset_jetPY[old_size:new_size] = jet_py
          dset_jetPZ[old_size:new_size] = jet_pz

          dset_E[old_size:new_size,:] = constituents[:,:,0]
          dset_PX[old_size:new_size,:] = constituents[:,:,1]
          dset_PY[old_size:new_size,:] = constituents[:,:,2]
          dset_PZ[old_size:new_size,:] = constituents[:,:,3]

          dset_ptfrac[old_size:new_size,:] = const_ptfrac
          dset_dphi[old_size:new_size,:] = const_dphi
          dset_deta[old_size:new_size,:] = const_deta
          dset_m[old_size:new_size,:] = const_m

          dset_kt[old_size:new_size,:] = ljp[:,:,0]
          dset_dr[old_size:new_size,:] = ljp[:,:,1]

          batch+=1
          Ncount+=stop_index-index

    #Close up and exit
    outfile_train.close()
    outfile_test.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process benchmarks.')
    parser.add_argument("filepaths", help="", nargs='+')
    parser.add_argument("--outname", help="", default="test.h5")
    parser.add_argument("--outdir", help="", default="inputFiles/")
    parser.add_argument("--format", choices=["topbenchmark","jetclass"], default="topbenchmark")
    parser.add_argument("--treename", help="", default="tree")
    parser.add_argument("--split",type=float, default=0.8, help="Train/test split fraction")
    parser.add_argument("--seed",type=float, default=99, help="Train/test split fraction")
    args = parser.parse_args()

    print(args.filepaths)

    #try:
    load_and_lundplane(args.filepaths, args.treename, outdir=args.outdir, outname=args.outname, train_test_split=args.split, seed=args.seed, file_format=args.format)
    #except Exception as e:
    #    print(f"Failed to read file {args.filename}: \n{e}")
