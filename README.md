# Transformer Lund

Transformer-based models for Lund-plane sequence generation. The current unbinned workflow is centered on `train_unbinned.py` and `plot_unbinned.py`. The older binned workflow is still available, but the examples below focus on the current scripts.

**Install Dependencies for CPU:**
```
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Install for GPU:**

```
conda create --name torch_env python=3.11
conda activate torch_env
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```


**Get dataset**
Input files for the top-benchmark dataset can be found here https://zenodo.org/records/2603256 and download via
```
mkdir inputFiles/top_benchmark
cd inputFiles/top_benchmark
wget https://zenodo.org/records/2603256/files/train.h5
wget https://zenodo.org/records/2603256/files/test.h5
wget https://zenodo.org/records/2603256/files/val.h5
cd ../..
```

Input files from the jetclass dataset from https://zenodo.org/records/6619768 can be downloaded via
```
python download_jetclass.py
```

**Data Setup**
To process the input root after download the Input Files:
```
source data_setup.sh
```
This calls the scripts `discretize.sh` (which is a wrapper around `preprocess.py`), `addEnergy.py`, `lunddata.py`, `lund_select.py`, and `discretize_auto.py` in sequence

`preprocess.py` runs the original paper pt/eta discretization and outputs to a /discretized directory of the original, and outputs a root tree of pt/eta/phi to `./originalJets_*.root`


Use `--logMode` for `lunddata.py` in `data_setup.sh` to output in log(1/deltaR) and log(kt) format, and remove `--logMode` to produce deltaR and kt. Add `--swapAxes` to switch the tree order of root file from deltaR and kt. This overwrites `originalJets_*.root`.
```
python lunddata.py --logMode
```

Use `lund_select.py` to cut the output. `--xmin --xmax` is the cut on kt, and `--ymin --ymax` is the cut on deltaR. Reads `inputFiles/qcd_lund.root` and outputs `inputFiles/qcd_lund_cut.root`.
```
python lund_select.py --in inputFiles/qcd_lund.root --out inputFiles/qcd_lund_cut.root --mode cut --xmin 0 --xmax 8 --ymin -1 --ymax 8
```
Could also use --mode top10 to select first 10 emissions, --mode shuffle to shuffle the order of emissions


To perform legacy binned training, the Lund input files must be discretized and written to H5 files in `inputFiles/discretized/`.

```
python discretize_auto.py --data_path inputFiles/qcd_lund_cut.root --nBins 41 31 --tag kt_deltaR --auto_const_q 0.9 --split_train_val --train_ratio 0.8
```

For the current unbinned workflow, `train_unbinned.py` expects an H5 file with either:

- `lundplane/dr` and `lundplane/kt` for `--input_format ktdr`
- `constituents/E`, `PX`, `PY`, `PZ` for `--input_format 4vec`

**Unbinned Training**

All current training modes share the same command structure:

```
python -u train_unbinned.py \
  --train-file inputFiles/jetclass.h5 \
  --val-file inputFiles/jetclass.h5 \
  --input_format ktdr \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.001 \
  --log-dir models/test
```

Common options:

- `--train-file`, `--val-file`: training and validation H5 files.
- `--input_format`: `ktdr` for Lund `dr/kt`, or `4vec` for constituent four-vectors.
- `--epochs`, `--batch-size`, `--lr`, `--patience`: core training controls.
- `--device`: force `cpu` or `cuda`; omitted means auto-detect.
- `--optimizer`: `adam` or `adamw`.
- `--weight-decay`: useful with `--optimizer adamw`.
- `--grad-clip`: gradient norm clipping; set `0` to disable.
- `--log-dir`: output directory. If it exists, a suffix like `_1` is added.

***Save Modes***

Training artifacts are written under `<log-dir>/checkpoints/`.

Default full checkpoints:

```
--save-mode checkpoint
```

This writes `epoch_XXX.pt` each epoch and `best.pt` for the best validation epoch. Full checkpoints include model state, optimizer state, scheduler state, args, losses, LR history, best epoch, and best loss.

Model-state only:

```
--save-mode model
```

This saves only `model_state_dict` plus metadata needed for plotting. It is smaller, but cannot fully resume optimizer/scheduler state.

No `.pt` artifacts:

```
--save-mode none
```

This does not write model artifacts, but still writes loss/LR plots and metadata.

Continue from a checkpoint:

```
python -u train_unbinned.py \
  --contin \
  --checkpoint models/test/checkpoints/best.pt \
  --train-file inputFiles/jetclass.h5 \
  --val-file inputFiles/jetclass.h5 \
  --epochs 200 \
  --log-dir models/test_resume
```

***Schedulers***

No scheduler:

```
--scheduler none
```

Cosine decay over epochs:

```
--scheduler cosine \
--scheduler-min-lr 1e-6
```

Reduce LR on plateau:

```
--scheduler plateau \
--plateau-factor 0.5 \
--plateau-patience 2 \
--scheduler-min-lr 1e-6
```

Cosine damping with optional oscillation:

```
--scheduler cos_damping \
--cos-damping-start-epoch 50 \
--cos-damping-end-epoch 300 \
--cos-damping-final-lr 5e-5 \
--cos-damping-amplitude 0.10 \
--cos-damping-period-epochs 5.0
```

**Training Modes**

***Regression***

Default mode. This trains a direct next-step regression head.

```
python -u train_unbinned.py \
  --train-file inputFiles/jetclass.h5 \
  --val-file inputFiles/jetclass.h5 \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.001 \
  --scheduler cosine \
  --optimizer adamw \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --log-dir models/regression
```

***MDN***

Mixture Density Network head. This is the current example mode in `transformer_run.sh`.

```
python -u train_unbinned.py \
  --train-file inputFiles/jetclass.h5 \
  --val-file inputFiles/jetclass.h5 \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.001 \
  --mdn \
  --n-mix 25 \
  --scheduler cosine \
  --optimizer adamw \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --log-dir models/mdn
```

***CNF***

Continuous Normalizing Flow head. CNF evaluation needs gradients during validation, so it is usually slower.

```
python -u train_unbinned.py \
  --train-file inputFiles/jetclass.h5 \
  --val-file inputFiles/jetclass.h5 \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.001 \
  --cnf \
  --cnf-hidden 128 \
  --cnf-steps 8 \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 2 \
  --grad-clip 1.0 \
  --log-dir models/cnf
```

**Batch Running**

For a batch job, edit `transformer_run.sh` with the desired `train_unbinned.py` command, then run or submit `transformer.sh`.

```
source transformer_setup.sh
source transformer_run.sh
```

`transformer.sh` sources `transformer_setup.sh` and `transformer_run.sh` in sequence.

**Plot Unbinned Results**

Use `plot_unbinned.py` with one or more checkpoint/model-state files. Checkpoints saved as `epoch_000.pt` are displayed in plot captions as `ep 1`; `best.pt` displays the best epoch using the same 1-based convention.

Plot one best checkpoint:

```
python plot_unbinned.py \
  --checkpoint models/mdn/checkpoints/best.pt \
  --train-file inputFiles/jetclass.h5 \
  --val-file inputFiles/jetclass.h5 \
  --input_format ktdr \
  --num-workers 0 \
  --plot-max-batches 100 \
  --plot-dir models/mdn_plots
```

Compare several runs:

```
python plot_unbinned.py \
  --checkpoint \
    models/regression/checkpoints/best.pt \
    models/mdn/checkpoints/best.pt \
    models/cnf/checkpoints/best.pt \
  --train-file inputFiles/jetclass.h5 \
  --val-file inputFiles/jetclass.h5 \
  --input_format ktdr \
  --num-workers 0 \
  --hist1d-ranges -5 6 -3 10 \
  --hist1d-bins 50 \
  --hist2d-bins 40 40 \
  --hist2d-shape 2 2 \
  --plot-dir models/comparison_plots
```

Main outputs include:

- `hist1d.png` / `hist1d.pdf`
- `hist1d_logy.png` / `hist1d_logy.pdf`
- `lund.png` / `lund.pdf`
- `loss_combined__*.png` / `loss_combined__*.pdf` when loss CSVs are available

**Legacy Binned Training**

The older binned workflow is still available through `train.py`.

```
python train.py \
  --num_epochs 5 \
  --num_features 2 \
  --num_bins 41 31 \
  --data_path inputFiles/discretized/qcd_lund_cut_lundTree_kt_deltaR_train.h5
```

To process legacy sampled results:

```
python sample_jets_auto.py --num_samples 10000 --model_dir models/test/
```

**Other Plotting Utilities**

***Input Files (`fileList.txt` example)***

One ROOT file path per line. Lines beginning with `#` are ignored.

```
/path/to/sample_A.root
#/path/to/subdir/sample_B.root
```

***plot.py***

Main Lund-plane plotter. Reads one or multiple ROOT files, extracts kt and deltaR or their logarithms, and produces 2D histograms.

```
python plot.py --file_list fileList.txt --zmin 0 --zmax 0.025 --maxN 10
```

***plot_1dhist.py***

Draws 1D histograms of scalar observables, such as jet pT, eta, or custom quantities. Use `--mode kdr` for kt/deltaR or `--mode kin` for pt/eta/phi.

```
python plot_1dhist.py --file_list fileList.txt --mode kdr --maxN 10
```

***plot_bhist.py***

Specialized binned histogram plotter for comparing distributions from multiple datasets or epochs.

```
python plot_bhist.py --file_list binList.txt
```

***plot_corr.py***

Computes and visualizes correlation matrices between variables.

```
python plot_corr.py --file_list fileList.txt
```

***plot_eem.py***

Emission-by-emission overlay plotter.

```
python plot_eem.py --file_list fileList.txt
```

***plot_loss.py***

Reads training-history ROOT files and plots loss vs epoch curves across multiple files.

```
python plot_loss.py --file_list fileList.txt
```

Loss function files can be found under `models/test` for legacy workflows.


**CITATION**

If this code is used for a scientific publication, please add the following citation:
```
@article{Finke:2023veq,
    author = {Finke, Thorben and Kr\"amer, Michael and M\"uck, Alexander and T\"onshoff, Jan},
    title = "{Learning the language of QCD jets with transformers}",
    eprint = "2303.07364",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1007/JHEP06(2023)184",
    journal = "JHEP",
    volume = "06",
    pages = "184",
    year = "2023"
}
```
