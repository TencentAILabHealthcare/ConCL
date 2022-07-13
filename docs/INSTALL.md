## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+
- PyTorch 1.1 or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.0/9.2/10.0/10.1
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2 (PyTorch-1.1 w/ NCCL-2.4.2 has a deadlock bug, see [here](https://github.com/open-mmlab/OpenSelfSup/issues/6))
- GCC(G++): 4.9/5.3/5.4/7.3

### Install openselfsup

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

c. Install other third-party libraries.

```shell
conda install faiss-gpu cudatoolkit=10.0 -c pytorch # optional for DeepCluster and ODC, assuming CUDA=10.0
```

d. Clone the openselfsup repository.

```shell
git clone https://github.com/open-mmlab/openselfsup.git
cd openselfsup
```

e. Install.

```shell
pip install -v -e .  # or "python setup.py develop"
```

f. Install Apex (optional), following the [official instructions](https://github.com/NVIDIA/apex), e.g.
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.

2. Following the above instructions, openselfsup is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.


### Prepare datasets

It is recommended to symlink your dataset root (assuming $YOUR_DATA_ROOT) to `$OPENSELFSUP/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

#### Prepare NCT-CRC-HE-100k dataset

- Download the NCT-CRC-HE-100K-NONORM dataset (the pre-training dataset) from the [link](https://zenodo.org/record/1214456#.YaRmkvHMJm8).
- Assuming that you usually store datasets in `$YOUR_DATA_ROOT`, you can create a folder `data` under `$OPENSELFSUP` and make a symlink `NCT`. Then move all images in NCT-CRC-HE-100K-NONORM to `data/NCT/data` folder.
- We provide the split used for training in `data/NCT/meta`.

At last, the folder looks like:

```
OpenSelfSup
├── openselfsup
├── benchmarks
├── configs
├── data
│   ├── NCT
│   │   ├── meta
│   │   |   ├── train.txt (for self-sup training, "filename\n" in each line)
│   │   |   ├── train_labeled.txt
│   │   |   ├── val.txt
│   │   |   ├── val_labeled.txt
│   │   ├── data
```

#### Prepare GlaS dataset and CRAG dataset

- You can download our pre-processed datasets from [GlaS-coco-format](https://drive.google.com/file/d/1xilq4FLMEs1CJKDfZWpBC2TKAxxIA17W/view?usp=sharing) and [CRAG-coco-format](https://drive.google.com/file/d/1ksQZ8y4xiTyPMDUijvWtYnM8afHx--u2/view?usp=sharing).