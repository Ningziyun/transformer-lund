source discretize.sh inputFiles/top_benchmark
python addEnergy.py
python lunddata.py --logMode
python lund_select.py --in inputFiles/qcd_lund.root --out inputFiles/qcd_lund_cut.root --mode cut --xmin 0 --xmax 8 --ymin -1 --ymax 8 --mode swap
