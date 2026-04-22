source discretize.sh inputFiles/top_benchmark
./mergeTrees.py originalJets_qcd.root originalJets_qcd_*.root
python addEnergy.py
python lunddata.py --logMode
#python lund_select.py --in inputFiles/qcd_lund.root --out inputFiles/qcd_lund_cut.root --mode cut --xmin -1 --xmax 8 --ymin -1 --ymax 8 --mode swap
python discretize_auto.py --data_path inputFiles/qcd_lund.root --nBins 41 31 --tag kt_deltaR --auto_const_q 1 --split_train_val --train_ratio 0.8
