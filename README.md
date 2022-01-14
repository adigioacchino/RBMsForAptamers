# Interpretable machine learning model for DNA aptamer design and analysis

This repository allows to reproduce the figures in [this paper](add_link_paper).

## Data:
%% The data used (described in the paper) has been uploaded to Zenodo at [this link](add_link_data).%% 

*The data used is (temporarily) dowloadable from [here](https://web.cubbit.io/link/#a3ba49a1-3ecc-46cc-a61d-aa7e96bf6a05) in compressed format. In future it will be provided through [Zenodo](https://zenodo.org/).*
For the notebooks to work, the 5 data files must be placed into a `data` folder, which must be created inside the main directory of this repository. 
In particular:
- files `s100_Nth.fasta` (where `N` is 5, 6, 7 or 8) are standard fasta files, and the descriptor of each sequence is of the form `seqX-Y`, where `X` is an increasing label, and `Y` is the number of times `seqX` has been obtained (number of counts of `seqX`).
- file `Aptamer_Exp_Results.csv` contains the sequences tested experimentally, with all the experimental results for each sequence.

## Dependencies:
The code to make the figures is in several Jupyter notebooks, all written in python 3. The notebooks use standard scientific computing packages, that can be obtained for instance through the [Anaconda distribution](https://www.anaconda.com/products/individual).
The code to instantiate, train and use Restricted Boltzmann Machines exploits the packages provided with [Jerome Tubiana's repository](https://github.com/jertubiana/PGM). The first cell of each notebook uses [`GitPython`](https://github.com/gitpython-developers/GitPython) to clone locally this repository. 

## Repository structure:
The notebooks to make the figures are in the main directory:
- `fitness.ipynb` can be used to reproduce Fig. 2 of the paper.
- `double_interpretation.ipynb` can be use to reproduce Fig. 3 of the paper; moreover, it can be used also to inspect any weight of RBM-D8, and to reproduce Suppl. Fig. S15a.
- `discriminator.ipynb` can  be used to reproduce Fig. 4 and Suppl. Fig. S14 of the paper.
- `single_interpretation.ipynb` can be used to reproduce Fig. 6 and Suppl. Fig. S15b of the paper.
- `discriminator_withcounts.ipynb` can  be used to reproduce Suppl. Fig. 13 of the paper.

The folder `RBM_retrained` contains the pre-trained RBMs used to do the figures of the paper.
The folder `RBM_training` contains a notebook, `RBMs_training.ipynb` that can be used to re-train any of the RBM used for the paper.
Finally, the folder `DNA_utils` contains 3 python packages with useful scripts to run the other notebooks (for instance, to load our DNA datasets in a format which is compatible with the RBM package).

## Contacts:
For comments or question, feel free to [contact me](mailto:andrea.dgioacchino@gmail.com).
