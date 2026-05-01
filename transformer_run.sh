python train_unbinned.py --train-file inputFiles/jetclass.h5 --val-file inputFiles/jetclass.h5
python plot_unbinned.py \
  --model-path models/test/model_0.pt \
  --train-file inputFiles/jetclass.h5 \
  --val-file inputFiles/jetclass.h5 \
  --input_format ktdr \
  --num-workers 0
