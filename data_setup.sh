#!/bin/bash 

source discretize.sh inputFiles/top_benchmark
./mergeTrees.py inputFiles/top_benchmark/rootified/originalJets_qcd.root inputFiles/top_benchmark/rootified/originalJets_qcd_*.root
python lunddata.py inputFiles/top_benchmark/rootified/originalJets_qcd.root --outname topbenchmark.h5 --format topbenchmark
#python lund_select.py --in inputFiles/qcd_lund.root --out inputFiles/qcd_lund_cut.root --mode cut --xmin -1 --xmax 8 --ymin -1 --ymax 8 --mode swap
#python discretize_auto.py --data_path inputFiles/qcd_lund_cut.root --nBins 41 31 --tag kt_deltaR --auto_const_q 0.9 --split_train_val --train_ratio 0.8

./lunddata.py inputFiles/jetclass/ZJetsToNuNu* --outname jetclass.h5 --format jetclass
