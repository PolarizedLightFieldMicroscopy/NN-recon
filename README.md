# NN-recon



## Installation
### Method 1
Create environment with
```
conda env create -f bir.yml
```
### Method 2

```
conda create -n nn-recon python=3.10 -y
conda activate nn-recon
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c jmcmurray os -y
conda install -c conda-forge matplotlib tifffile -y
conda install -c anaconda ipykernel -y
pip install torchsummary
pip install torchview
conda install -c conda-forge torchinfo -y
pip install graphviz
conda install -c conda-forge tensorboard -y
```
**Notes**

Installing `graphviz` with conda as `conda install -c anaconda graphviz` caused the module `graphviz` not to be found. Installing with pip worked.