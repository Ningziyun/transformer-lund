#!/usr/bin/env python

import h5py
import hdf5plugin
import sys,os
import numpy as np
import matplotlib.pyplot as plt

def make_hist(valueslist,name,Nbins=20):
  fig, ax = plt.subplots()
  
  minval=np.min([np.min(vals) for vals in valueslist])
  maxval=np.max([np.max(vals) for vals in valueslist])
  #print(np.histogram_bin_edges(valueslist[0]))
  bins=np.linspace(minval,maxval,Nbins)

  return_contents=[]
  for values in valueslist:
    contents,bins,_=ax.hist(values,bins=bins,histtype="step")
    return_contents.append(contents)
  #plt.show()
  ax.set_yscale("log")
  fig.savefig(name)
  plt.close(fig)

  return return_contents, bins

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
        data = val[()]
        filledval = data[data != -1]
        print(f"{space}\t mean={np.mean(filledval):.3f}, std={np.std(filledval):.3f}")
        print(f"{space}\t min={np.min(filledval):.3f}, max={np.max(filledval):.3f}")
        #print(f"{space}\t {val[:]}")
    elif isinstance(val, h5py.Group):
      print(f"{space} Group: {key}")
      recursivePrint(val,depth+1)
    else: 
      print(f"{space} Unknown {key}: {val}") 
  
def recursiveDraw(f, N=1000, groupname=""):
  if not os.path.exists("./Plots"): os.mkdir("./Plots")

  for key,val in f.items():
    if isinstance(val, h5py.Dataset):
      print(f"Drawing {N} entries of key {key} in group {groupname}")
      print(f"{val[:100]}")
      values=val[:N].flatten()
      make_hist([values],"Plots/plot_"+groupname+key+".pdf")
    elif isinstance(val, h5py.Group):
      recursiveDraw(val,N,groupname=key+"_")

def findValByKey(f,key):
    for key2,val in f.items():
        if isinstance(val, h5py.Dataset):
            if key2==key: 
                return val
        elif isinstance(val, h5py.Group):
            return findValByKey(val,key)
    return None

def deriveFlattening(f,key):
    values=findValByKey(f,key)
    values=values[:].flatten()
    #values=values[values!=-1]
    content,bins=make_hist([values],"Plots/plot_"+key+".pdf",50)

    weights=1/content[0]
    weights[np.isinf(weights)] = 0
    print("weights",weights)
    print("bin_edges=",bins)

if __name__ == "__main__":
  if len(sys.argv)<2:
    print("No input files")
    sys.exit()

  infile=sys.argv[1]
  f = h5py.File(infile, 'r')

  #recursivePrint(f)
  recursiveDraw(f,100)
  #deriveFlattening(f,"E")

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
