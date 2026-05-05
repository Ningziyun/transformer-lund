#!/usr/bin/env python3

import os
import re
import glob
import subprocess
import requests
from tqdm import tqdm

def _remote_size(url):
    resp = requests.head(url, allow_redirects=True, timeout=30)
    resp.raise_for_status()
    return int(resp.headers.get("content-length", 0))

def _content_range_total(resp):
    match = re.match(r"bytes \d+-\d+/(\d+)", resp.headers.get("content-range", ""))
    return int(match.group(1)) if match else 0

def _download(url, fname, chunk_size=1024 * 1024, retries=5):
    part_name = f"{fname}.part"
    remote_size = _remote_size(url)

    if os.path.exists(fname):
        local_size = os.path.getsize(fname)
        if remote_size and local_size == remote_size:
            return
        os.replace(fname, part_name)

    for attempt in range(1, retries + 1):
        if remote_size and os.path.exists(part_name) and os.path.getsize(part_name) > remote_size:
            print(f"{part_name} is larger than the remote file; restarting download")
            os.remove(part_name)

        resume_from = os.path.getsize(part_name) if os.path.exists(part_name) else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as resp:
                resp.raise_for_status()

                if resume_from and resp.status_code != 206:
                    print(f"{fname}: server did not accept resume; restarting download")
                    resume_from = 0
                    mode = "wb"
                else:
                    mode = "ab" if resume_from else "wb"

                total = _content_range_total(resp) or remote_size or int(resp.headers.get("content-length", 0))
                with open(part_name, mode) as file, tqdm(
                    desc=fname,
                    total=total,
                    initial=resume_from,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in resp.iter_content(chunk_size=chunk_size):
                        if not data:
                            continue
                        size = file.write(data)
                        bar.update(size)
        except requests.RequestException as exc:
            if attempt == retries:
                raise RuntimeError(f"Download failed for {fname}") from exc
            print(f"{fname}: download interrupted ({exc}); retrying {attempt}/{retries}")
            continue

        if not remote_size or os.path.getsize(part_name) == remote_size:
            break
        if attempt == retries:
            local_size = os.path.getsize(part_name)
            raise RuntimeError(
                f"Incomplete download for {fname}: got {local_size} bytes, expected {remote_size} bytes"
            )
        print(f"{fname}: incomplete download; retrying {attempt}/{retries}")

    os.replace(part_name, fname)

def unpack(url):
  fname=url.split("/")[-1]
  done=f"{fname}.done"
  if os.path.exists(done): return
  _download(url,fname)
  subprocess.call(["tar","-xf",fname])
  if "train" in fname:
    subprocess.call(" ".join(["rm","HTo*root","TTBarLep*root","*ToQQ_*.root"]),shell=True)
  else:
    subprocess.call(" ".join(["rm","*/HTo*root","*/TTBarLep*root","*/*ToQQ_*.root"]),shell=True)
    if "test" in fname:
        subprocess.call(" ".join(["mv","test_20M/*","../"]),shell=True)
        subprocess.call(" ".join(["rm", "-r","test_20M"]))
    if "val" in fname:
        subprocess.call(" ".join(["mv","val_5M/*","../"]),shell=True)
        subprocess.call(" ".join(["rm", "-r","val_20M"]))
  '''
  for pattern in ["HTo*root", "TTBarLep*root", "*ToQQ_*.root"]:
    for path in glob.glob(pattern):
      os.remove(path)
  '''
  open(done, "w").close()

if __name__ == "__main__":

  cwd = os.getcwd()
  path="./inputFiles/jetclass/"
  os.makedirs(path, exist_ok=True)
  os.chdir(path)

  for ii in range(10):
    unpack(f"https://zenodo.org/records/6619768/files/JetClass_Pythia_train_100M_part{ii}.tar")
  unpack("https://zenodo.org/records/6619768/files/JetClass_Pythia_test_20M.tar")
  unpack("https://zenodo.org/records/6619768/files/JetClass_Pythia_val_5M.tar")

  subprocess.call(" ".join(["rm","*.tar"]))
  os.chdir(cwd)
