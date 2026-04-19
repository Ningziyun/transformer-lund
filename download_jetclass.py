#!/usr/bin/env python3

import subprocess,os
import requests
from tqdm import tqdm

def _download(url, fname, chunk_size=1024):
    '''Copied mostly from https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51'''
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def unpack(url):
  fname=url.split("/")[-1]
  if os.path.exists(fname): return
  _download(url,fname)
  subprocess.call(["tar","-xf",fname])
  subprocess.call(" ".join(["rm","HTo*root","TTBarLep*root","*ToQQ_*.root"]),shell=True)


if __name__ == "__main__":

  cwd = os.getcwd()
  path="./inputFiles/jetclass/"
  os.makedirs(path, exist_ok=True)
  os.chdir(path)

  for ii in range(1,10):
    unpack(f"https://zenodo.org/records/6619768/files/JetClass_Pythia_train_100M_part{ii}.tar")

  os.chdir(cwd)
