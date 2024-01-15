# HCGNet

Yiqun Lin, Liang Pan, Yi Li, Ziwei Liu, and Xiaomeng Li, "Exploiting Hierarchical Interactions for Protein Surface Learning," J-BHI 2024.

## 0. Citation

TODO.

## 1. Installation

python 3.6, CUDA 11.1

```shell
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm msgpack six tabulate termcolor pyyaml easydict
pip install Biopython sklearn ninja==1.10.2
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

cd pointnet2
python setup.py install
```

## 2. Data Preparation

Data can be downloaded and processed following https://github.com/FreyrS/dMaSIF/blob/master/data.py. The raw data is structured as

```
./data/raw/
    ├── 01-benchmark_pdbs
    │   └── 1A0G_A.pdb
    ├── 01-benchmark_surfaces
    │   └── 1A0G_A.ply
```

Then, modify the path (`DATA_RAW`) in `./utils/config.py` to the data folder. For each task (site/search), run the preprocessing script (`./<pdb_task>/preprocessing.py`) to generate training/testing data.

## 3. Training and Testing

For each task (site/search), follow the scripts given in `./tasks/<pdb_task>/scripts/<train/test>.sh` to conduct training and testing.


## License

This repository is released under MIT License (see LICENSE file for details).
