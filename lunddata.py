#!/usr/bin/env python3

import argparse
import os, sys, glob
import ROOT
import uproot
import awkward as ak
import numpy as np
import h5py

from helpers_unbinned import *

def loopTree(files, treename, outdir="inputFiles/", outname="qcd_lund.root", logMode=False, mode="kt", fileformat="topbenchmark"):

    #make the output file
    if not os.path.exists(outdir):
      os.makedirs(outdir)
    outpath = os.path.join(outdir, outname)
    outfile=h5py.File(outpath, "w")

    #Set the padding size
    Npad_const=100
    Npad_ljp=20

    #Loop over input files
    for nfile,filename in enumerate(files):

      #Get the tree via either old root or uproot
      if opt.format=="topbenchmark":
        file = ROOT.TFile(filename)
        tree = file.Get(treename)
      elif opt.format=="jetclass":
        file = uproot.open(filename)
        tree = file[treename]

      if not tree:
        continue
      print(f"Running file {filename} and tree {treename}")
       
      if fileformat=="topbenchmark":

        #If first file make the hdf5 format, fixing the size to tree length since will be 1 file
        if nfile==0:
          grp_const = outfile.create_group("constituents")
          dset_E = grp_const.create_dataset("E", shape=(tree.GetEntries(),Npad_const), dtype="float32")
          dset_PX = grp_const.create_dataset("PX", shape=(tree.GetEntries(),Npad_const), dtype="float32")
          dset_PY = grp_const.create_dataset("PY", shape=(tree.GetEntries(),Npad_const), dtype="float32")
          dset_PZ = grp_const.create_dataset("PZ", shape=(tree.GetEntries(),Npad_const), dtype="float32")

          grp_ljp = outfile.create_group("lundplane")
          dset_kt = grp_ljp.create_dataset("kt", shape=(tree.GetEntries(),Npad_ljp), dtype="float32")
          dset_dr = grp_ljp.create_dataset("dr", shape=(tree.GetEntries(),Npad_ljp), dtype="float32")

        # Loop through all events one at at time and store jet/const info
        for index, event in enumerate(tree):
          #make an array size [1, Nconst, 4-vector]
          constituents=np.zeros([1,Npad_const,4])
          if index%1000 == 0: print("index",index)
          
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
          dset_E[index,:] =  constituents[0,:,0]
          dset_PX[index,:] = constituents[0,:,1]
          dset_PY[index,:] = constituents[0,:,2]
          dset_PZ[index,:] = constituents[0,:,3]

          dset_kt[index,:] = ljp[0,:,0]
          dset_dr[index,:] = ljp[0,:,1]

      if fileformat=="jetclass":

        #If first file make the hdf5 format, will leave it empty in length and fill it up as we go on-the-fly
        if nfile==0:
          grp_const = outfile.create_group("constituents")
          dset_E = grp_const.create_dataset("E", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PX = grp_const.create_dataset("PX", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PY = grp_const.create_dataset("PY", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")
          dset_PZ = grp_const.create_dataset("PZ", shape=(0,Npad_const), maxshape=(None, Npad_const), dtype="float32")

          grp_ljp = outfile.create_group("lundplane")
          dset_kt = grp_ljp.create_dataset("kt", shape=(0,Npad_ljp), maxshape=(None, Npad_const), dtype="float32")
          dset_dr = grp_ljp.create_dataset("dr", shape=(0,Npad_ljp), maxshape=(None, Npad_const), dtype="float32")

        #Loop over the input files in chunks via uproot
        chunk_size = 2**11
        batch=0
        for index in range(0, tree.num_entries, chunk_size): #loading in 1 event at a time, hence some [0] below
          print(f"batch={batch},index={index}")
          stop_index=min(index + chunk_size, tree.num_entries)

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
          batch+=1

    #Close up and exit
    outfile.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process benchmarks.')
    parser.add_argument("filepaths", help="", nargs='+')
    parser.add_argument("--outname", help="", default="test.h5")
    parser.add_argument("--format", choices=["topbenchmark","jetclass"], default="topbenchmark")
    parser.add_argument("--treename", help="", default="tree")
    parser.add_argument("--logMode", action="store_true", help="If set, output log(kt or z) and log(1/deltaR)")
    parser.add_argument("--swapAxes", action="store_true", help="If set, swap the order of kt and deltaR in output")
    parser.add_argument("--mode", type=str, choices=["kt", "z"], default="kt", help="Select variable to pair with deltaR: 'kt' (default) or 'z'")
    opt = parser.parse_args()

    print(opt.filepaths)

    #try:
    loopTree(opt.filepaths, opt.treename, outname=opt.outname, logMode=opt.logMode, mode=opt.mode, fileformat=opt.format)
    #except Exception as e:
    #    print(f"Failed to read file {opt.filename}: \n{e}")
