python train.py  --num_epochs 100 --lr 0.0005 --num_features 2 --num_bins 100 100  --data_path inputFiles/kt_deltaR.h5
python sample_jets_auto.py --num_samples 10000 --model_dir models/test_5/
