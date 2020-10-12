
## Setup

Instructions for set up on Ubuntu 18.04 with Miniconda and Python 3.7

* Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* Create a conda env for the project
```
conda create  -y -n seq_des python=3.7 anaconda
conda activate seq_des
```
* Install [PyRosetta4](http://www.pyrosetta.org/dow) via conda
  * Get a license for PyRosetta
  * Add this to ~/.condarc
  ```
  channels:
    - https://USERNAME:PASSWORD@conda.graylab.jhu.edu
    - conda-forge
    - defaults
  ```
  * Install PyRosetta
  ```
  conda install pyrosetta
  ```
* Install PyTorch 1.1.0 with CUDA 9.0
```
conda install -y pytorch=1.1.0 torchvision cudatoolkit=9.0 -c pytorch
```
* Clone this repo
```
git clone https://github.com/nanand2/protein_seq_des.git
```
* Install Python packages
```
cd protein_seq_des
pip install -r requirements.txt
```

* Download [pretrained models](https://drive.google.com/file/d/1X66RLbaA2-qTlJLlG9TI53cao8gaKnEt/view?usp=sharing) to current directory

```
unzip models.zip
```

