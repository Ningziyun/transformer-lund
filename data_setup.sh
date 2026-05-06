#!/bin/bash 

mkdir -p inputFiles/top_benchmark
cd inputFiles/top_benchmark
wget https://zenodo.org/records/2603256/files/train.h5
wget https://zenodo.org/records/2603256/files/test.h5
wget https://zenodo.org/records/2603256/files/val.h5
cd ../..
source discretize.sh inputFiles/top_benchmark
python lunddata.py inputFiles/top_benchmark/rootified/originalJets_qcd_*.root --outname topbenchmark.h5 --format topbenchmark

python download_jetclass.py
./lunddata.py inputFiles/jetclass/ZJetsToNuNu* --outname jetclass.h5 --format jetclass
