**Install Dependencies for CPU:**
```
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Install for GPU:**

```
conda create --name torch_env python=3.9
conda activate torch_env
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

**Discretizing**

To perform the training, the input files must be discretized. 
This can be performed by running the discretization script, where the first argument is the path to the original root files.
To produce the qcd_lund.root file after download the Input files, process the code in Data Setup part.

```
Python discretize_auto.py --data_path inputFiles/qcd_lund.root --nBins 41 31 --tag kt_deltaR --auto_const_q 0.9 --split_train_val --train_ratio 0.8

```



To train a model run:
```
python train.py  --num_epochs 5 --num_features 2 --num_bins 41 31 --data_path inputFiles/discretized/qcd_lund_lundTree_kt_deltaR_train.h5
```

To process the results:

```
python sample_jets_auto.py --num_samples 10000 --model_dir models/test/
```


**Input Files**


Input files for this code can be found at https://zenodo.org/records/2603256


**Data Setup**


To process the input root after download the Input Files:
```
source date_setup.sh
```

Use --logMode in data_detup.sh to output in log(1/deltaR) and log(kt) format, and deleta --logMode to produce deltaR and kt.
```
python lunddata.py --logMode
```

Add --swapAxes to switch the tree order of root file from deltaR and kt to kt and deltaR.
```
python lunddata.py --swapAxes
```
**CITATION**

If this code is used for a scientfic publication, please add the following citation
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
