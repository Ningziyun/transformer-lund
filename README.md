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
This can be performed by running the discretization script, where the first argument is the path to the original h5 files.

```
source discretize.sh inputFiles/top_benchmark
```



To train a model run:
```
python train.py  --data_path inputFiles/top_benchmark/discretized/train_qcd_pt80_eta60_phi60_lower001.h5
```

To process the results:

```
python sample_jets.py --model_dir models/test_1/
```


**Input Files**


Input files for this code can be found at https://zenodo.org/records/2603256


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
