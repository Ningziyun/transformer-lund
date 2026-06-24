#!/usr/bin/env python3

import argparse
import os, sys, glob
import ROOT
import uproot
import awkward as ak
import numpy as np
import random
import h5py

from helpers_unbinned import *

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

          #Add to file
          '''
          if random.random()<train_test_split:
              dset_E_train[index,:] =  constituents[0,:,0]
              dset_PX_train[index,:] = constituents[0,:,1]
              dset_PY_train[index,:] = constituents[0,:,2]
              dset_PZ_train[index,:] = constituents[0,:,3]

              dset_kt_train[index,:] = ljp[0,:,0]
              dset_dr_train[index,:] = ljp[0,:,1]
          else:
              dset_E_test[index,:] =  constituents[0,:,0]
              dset_PX_test[index,:] = constituents[0,:,1]
              dset_PY_test[index,:] = constituents[0,:,2]
              dset_PZ_test[index,:] = constituents[0,:,3]

              dset_kt_test[index,:] = ljp[0,:,0]
              dset_dr_test[index,:] = ljp[0,:,1]
          '''
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
          dset_kt[-constituents.shape[0]:,:] = ljp[:,:,1]
          dset_dr[-constituents.shape[0]:,:] = ljp[:,:,0]
          Ncount+=1

      if file_format=="jetclass":

        #If first file make the hdf5 format, will leave it empty in length and fill it up as we go on-the-fly
        if nfile==0:
          grp_const_train = outfile_train.create_group("constituents")
          dset_E_train = grp_const_train.create_dataset("E", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PX_train = grp_const_train.create_dataset("PX", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PY_train = grp_const_train.create_dataset("PY", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PZ_train = grp_const_train.create_dataset("PZ", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")

          grp_ljp_train = outfile_train.create_group("lundplane")
          dset_kt_train = grp_ljp_train.create_dataset("kt", shape=(0,Npad_ljp), maxshape=(None, Npad_const), dtype="float32")
          dset_dr_train = grp_ljp_train.create_dataset("dr", shape=(0,Npad_ljp), maxshape=(None, Npad_const), dtype="float32")

          grp_const_test = outfile_test.create_group("constituents")
          dset_E_test = grp_const_test.create_dataset("E", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PX_test = grp_const_test.create_dataset("PX", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PY_test = grp_const_test.create_dataset("PY", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PZ_test = grp_const_test.create_dataset("PZ", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")

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
          unpacked = tree.arrays(["part_energy","part_px", "part_py", "part_pz"], entry_start=index, entry_stop=stop_index, library="np")
          Njet=len(unpacked["part_energy"])

          #Make the empty array we will fill
          constituents=np.zeros([Njet,Npad_const,4])
          
          #Now fill it with the 4-vectors
          for ii in range(Njet):
            for jj in range(min(len(unpacked["part_energy"][ii]),Npad_const)):
              constituents[ii,jj,:]=[unpacked["part_energy"][ii][jj],unpacked["part_px"][ii][jj],unpacked["part_py"][ii][jj],unpacked["part_pz"][ii][jj]]

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
          dset_kt[-constituents.shape[0]:,:] = ljp[:,:,1]
          dset_dr[-constituents.shape[0]:,:] = ljp[:,:,0]
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
    parser.add_argument("--format", choices=["topbenchmark","jetclass"], default="topbenchmark")
    parser.add_argument("--treename", help="", default="tree")
    parser.add_argument("--split",type=float, default=0.8, help="Train/test split fraction")
    parser.add_argument("--seed",type=float, default=99, help="Train/test split fraction")
    args = parser.parse_args()

    print(args.filepaths)

    #try:
    load_and_lundplane(args.filepaths, args.treename, outname=args.outname, train_test_split=args.split, seed=args.seed, file_format=args.format)
    #except Exception as e:
    #    print(f"Failed to read file {args.filename}: \n{e}")
