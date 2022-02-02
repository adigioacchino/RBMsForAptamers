# Interpretable machine learning model for DNA aptamer design and analysis

This repository allows to reproduce the figures in [this paper](add_link_paper).
This repository also contains the DNN and traditional ML models used for comparison with the RBM. No explicit script to generate the DNN or trad ML model figures is included.

## Data:
%% The data used (described in the paper) has been uploaded to Zenodo at [this link](add_link_data).%%

*The data used is (temporarily) dowloadable from [here](https://web.cubbit.io/link/#a3ba49a1-3ecc-46cc-a61d-aa7e96bf6a05) in compressed format. In future it will be provided through [Zenodo](https://zenodo.org/).*
For the notebooks to work, the 5 data files must be placed into a `data` folder, which must be created inside the main directory of this repository.
In particular:
- files `s100_Nth.fasta` (where `N` is 5, 6, 7 or 8) are standard fasta files, and the descriptor of each sequence is of the form `seqX-Y`, where `X` is an increasing label, and `Y` is the number of times `seqX` has been obtained (number of counts of `seqX`).
- file `Aptamer_Exp_Results.csv` contains the sequences tested experimentally, with all the experimental results for each sequence.

### Please note: **Raytune DOES NOT support relative paths to data files. To enable ray tune optimization (for DNN models only), Line 207 and Line 208 of `MLmodels.py' (in DNN_Models directory) must be changed to the absolute paths to the data folder and DNN_helper_files folder.**


## Dependencies:
The code to make the figures is in several Jupyter notebooks, all written in python 3. The notebooks use standard scientific computing packages, that can be obtained for instance through the [Anaconda distribution](https://www.anaconda.com/products/individual).
The code to instantiate, train and use Restricted Boltzmann Machines exploits the packages provided with [Jerome Tubiana's repository](https://github.com/jertubiana/PGM). The first cell of each notebook uses [`GitPython`](https://github.com/gitpython-developers/GitPython) to clone locally this repository.

To use the code for the DNN models or other traditional machine learning models, a yml file `exmachina.yml` is included in directory DNN_Models that will allow one to build an anaconda env to run the code.
## Repository structure:
The notebooks to make the figures are in the main directory:
- `fitness.ipynb` can be used to reproduce Fig. 2 of the paper.
- `double_interpretation.ipynb` can be use to reproduce Fig. 3 of the paper; moreover, it can be used also to inspect any weight of RBM-D8, and to reproduce Suppl. Fig. S17a.
- `discriminator.ipynb` can  be used to reproduce Fig. 4 and Suppl. Fig. S16 of the paper.
- `single_interpretation.ipynb` can be used to reproduce Fig. 6 and Suppl. Fig. S17b of the paper.
- `discriminator_withcounts.ipynb` can  be used to reproduce Suppl. Fig. S8 of the paper.

The folder `RBM_retrained` contains the pre-trained RBMs used to do the figures of the paper.
The folder `RBM_training` contains a notebook, `RBMs_training.ipynb` that can be used to re-train any of the RBM used for the paper, and to produce Suppl. Fig. 18 of the paper.
The folder `DNA_utils` contains 3 python packages with useful scripts to run the other notebooks (for instance, to load our DNA datasets in a format which is compatible with the RBM package).

### The directory `DNN_Models` contains:

5 DNN models as pytorch lightning modules in the files `VAE_lightning_long.py`, `VAE_lightning_short.py`, `siamese_lightning.py`, `resnet_lightning.py`, and `resnet_fc_lightning.py`.

    The main call of each model contains a debugging loop, a pytorch lightning run, and one or more raytune hyperparameter optimization calls.
    Be sure to check the necessary arguments of each script before using. Below is an example call for raytune optimization using siamese_lightning.py

        python siamese_lightning.py HCLT 1 1 1 1
        where arg1 = dataset, arg2 = gpus_per_trial, arg3 = cpus_per_trial,  arg4 = samples

        Our dataset is represented by a 4-7 character string which is passed to our get_rawdata function in `MLModels.py`
        **Raytune DOES NOT support relative paths to data files. At least not that I've figured out. To enable ray tune, Line 207 of `MLmodels.py`
        must be changed to the absolute path to the data directory. Line 208 of `MLmodels.py` must be changed to the absolute path to the
        DNN_helper_files folder.**

             The six datasets (all from the 8th round) used in this work were:
                HCLT, HCRT, HCB20T, HCGLT, HCGRT, HCGB20T

        The gpus argument sets the number of gpus to use per sample. If < number of available gpus, multiple samples will run concurrently.
        The cpus argument sets the number of cpus to use per sample. If < number of available cpus, multiple samples will run concurrently.
        The samples argument tells raytune how many samples of the hyperparameter space it should perform.

`exmachina.yml` - a yml file for creating an anaconda environment with the necessary dependencies for running models in the DNN_Models and Trad_Models directories
`MLmodels.py` - contains many universal functions/modules used by the individual pytorch lightning scripts.
`result_collector.py` - contains functions for prediction and figure creation for all models in the DNN directory

### The directory `DNN_helper_files` contains:

`fakesequences.csv` - Randomly generated sequences used as bad binders for DNN training if "G" is in the dataset string
`round_8_nn.csv` - Hamming Distance of nearest neighbor sequence, used to remove "passenger" strands by including "T" in the dataset string.

### The directory `Trad_Models` includes:

`HCT_forest.ipynb`, `HCGT_forest.ipynb` - Jupyter Notebooks for single decision tree, random forest, and gradient boosted classification tree on each of the six datasets used in the paper.
`MLmodels.py` - an exact copy of the one in DNN_Models, only included for the data import functions

## Contacts:
For comments or questions on the RBM training and associated figures, feel free to [contact Andrea](mailto:andrea.dgioacchino@gmail.com).
For comments or questions on the DNN or Traditional Model training/usage, feel free to [contact Jonah](mailto:jprocyk@asu.edu).
