#!/usr/bin/env python

import h5py
import hdf5plugin
import sys,os
import numpy as np
import matplotlib.pyplot as plt

def make_hist(valueslist,name):
  fig, ax = plt.subplots()

  
  minval=np.min([np.min(vals) for vals in valueslist])
  maxval=np.max([np.max(vals) for vals in valueslist])
  #print(np.histogram_bin_edges(valueslist[0]))
  bins=np.linspace(minval,maxval,20)

  for values in valueslist:
    ax.hist(values,bins=bins,histtype="step")
  #plt.show()
  fig.savefig(name)
  plt.close(fig)

def recursivePrint(f,depth=0):
  space="\t"*depth
  if depth==0: print(f"{f}")
  for key,val in f.attrs.items():
    if isinstance(val, np.bytes_): val=val.decode('UTF-8')
    print(f"{space} Attribute {key}: {val}")
  for key,val in f.items():
    if isinstance(val, h5py.Dataset):
        print(f"{space} Dataset: {key}")
        print(f"{space}\t type={val.dtype}, shape={val.shape}")
        #print(f"{space}\t {val[:]}")
    elif isinstance(val, h5py.Group):
      print(f"{space} Group: {key}")
      recursivePrint(val,depth+1)
    else: 
      print(f"{space} Unknown {key}: {val}") 
  
def recursiveDraw(f,N=1000):
  if not os.path.exists("./Plots"): os.mkdir("./Plots")

  for key,val in f.items():
    if isinstance(val, h5py.Dataset):
      print(f"{key}\t {val[:100]}")
      values=val[:N].flatten()
      make_hist([values],"Plots/plot_"+key+".pdf")
    elif isinstance(val, h5py.Group):
      recursiveDraw(val,N)

if __name__ == "__main__":
  if len(sys.argv)<2:
    print("No input files")
    sys.exit()

  infile=sys.argv[1]
  f = h5py.File(infile, 'r')

  recursivePrint(f)
  recursiveDraw(f,100)

  #Draw seperating via labels, hardcoded right now
  '''
  if not os.path.exists("./Plots"): os.mkdir("./Plots")
  #get lables
  labels=f["labels"][:Njet]

  # loop on collections
  for var in f.keys():
    print(var)
    #if f[var].ndim>1: continue
    #values=f[var][:Njet]
    values_sig=[]
    values_bkg=[]
    for index,label in enumerate(labels):
      if label==1:
        values_sig.append(f[var][index])
      if label==0:
        values_bkg.append(f[var][index])
    values_sig=np.asarray(values_sig).flatten()
    values_bkg=np.asarray(values_bkg).flatten()
    make_hist([values_sig,values_bkg],"Plots/plot_"+var+".pdf")
    '''
