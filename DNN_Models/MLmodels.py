import pandas as pd
import glob, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import combinations, permutations
from torch.utils.data import Dataset
import os

import torch
import multiprocessing as mp

from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import sklearn

import matplotlib.pyplot as plt
import torchvision.models as models

from torch.optim import lr_scheduler as scheduler
import torch.optim as optim

from sklearn.model_selection import train_test_split
from typing import List
import math

from collections import defaultdict

import seaborn as sn
import pandas as pd

from torch.nn import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm as tqdm_base
#from tqdm.auto import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)



### Fancy Annealing

class CosineAnnealingLR_with_Restart(_LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. The original pytorch
    implementation only implements the cosine annealing part of SGDR,
    I added my own implementation of the restarts part.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        T_mult (float): Increase T_max by a factor of T_mult
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        model (pytorch model): The model to save.
        out_dir (str): Directory to save snapshots
        take_snapshot (bool): Whether to save snapshots at every restart
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_mult, model, out_dir, take_snapshot, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.Te = self.T_max
        self.eta_min = eta_min
        self.current_epoch = last_epoch

        self.model = model
        self.out_dir = out_dir
        self.take_snapshot = take_snapshot

        self.lr_history = []

        super(CosineAnnealingLR_with_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lrs = [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.current_epoch / self.Te)) / 2 for base_lr in self.base_lrs]

        self.lr_history.append(new_lrs)
        return new_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        ## restart
        if self.current_epoch == self.Te:
            #print("restart at epoch {:.5f}".format(self.last_epoch + 1))

            #if self.take_snapshot:
            #    torch.save(
            #        {"epoch": self.T_max, "state_dict": self.model.state_dict()}, self.out_dir + "/" + "snapshot_e_{:.5f}.pth.tar".format(self.T_max)
            #    )

            ## reset epochs since the last reset
            self.current_epoch = 0

            ## reset the next goal
            self.Te = int(self.Te * self.T_mult)
            self.T_max = self.T_max + self.Te

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))


# Fasta File Reader, with affinities
def fasta_read(fastafile, seq_read_counts=False, drop_duplicates=False):
    o = open(fastafile)
    titles = []
    seqs = []
    for line in o:
        if line.startswith('>'):
            if seq_read_counts:
                titles.append(float(line.rstrip().split('-')[1]))
        else:
            seqs.append(line.rstrip())
    o.close()
    if drop_duplicates:
        # seqs = list(set(seqs))
        all_seqs = pd.DataFrame(seqs).drop_duplicates()
        seqs = all_seqs.values.tolist()
        seqs = [j for i in seqs for j in i]

    if seq_read_counts:
        return seqs, titles
    else:
        return seqs


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def init_torch():
    is_cuda = torch.cuda.is_available()
    device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
    if is_cuda:
        torch.backends.cudnn.benchmark = True

    print(f'Preparing to use device {device}')
    random.seed(5000)
    # torch.manual_seed(5000)

def get_rawdata(type, countcutoff, n_copies, round=8, outputnum=2, seed=69):
    '''
    Loads the dataset of choice and prepares data

    :param round: selex round (6, 7, or 8)
    :param dtype: corresponds to subset of data you want to look at
    options are: both20M, both20, both40, left, right
    :param rtype: left, right, both20, both40
    :param countcutoff: the cutoff value (affinity or inferred count) designating a good binder
    :param n_copies: bad binders in dataset = (number of good binders) * n_copies
    :return: (train, validate) datasets should be ready for processing by pytorch Dataloaders
    '''

    # path to data, this works for pytorch lightning loops, but not for raytune
    rel_path = "../data/"

    # FOR RAYTUNE TO WORK YOU MUST PROVIDE AN ABSOLUTE PATH TO THE DATA HERE
    # ex.  rel_path = "/home/$USER/RBMsForAptamers/

    if round in [5, 6, 7, 8]:
        fasta_file = "s100_" + str(round) + "th.fasta"
        seqs, affs = fasta_read(rel_path + fasta_file, seq_read_counts=True, drop_duplicates=True)
        seq_l = [s[:20] for s in seqs]
        seq_r = [s[20:] for s in seqs]

        seq_l, c_l = np.unique(seq_l, return_counts=True, axis=0)
        seq_r, c_r = np.unique(seq_r, return_counts=True, axis=0)

        c_ldict = dict(zip(seq_l, c_l))
        c_rdict = dict(zip(seq_r, c_r))

        count_l = [c_ldict[s] for s in seq_l]
        count_r = [c_rdict[s] for s in seq_r]
    else:
        print(f"Round {round} not supported")
        exit(1)

    if round in [5, 6, 7]:
        if 'C' in type:  # normal unadjusted data
            if 'L' in type:
                data = pd.DataFrame({'sequence': seq_l, 'affinity': count_l})
            if 'R' in type:
                data = pd.DataFrame({'sequence': seq_r, 'affinity': count_r})
            if 'B20' in type:
                data = pd.DataFrame({'sequence': seq_l + seq_r, 'affinity': count_l + count_r})
            elif 'B' in type:
                data = pd.DataFrame({'sequence': seqs, 'affinity': affs})

    elif round == 8:
        # Helper File
        tmp = pd.read_csv(rel_path+"round_8_nn.csv")
        nn_l = tmp["nn_l"].tolist()
        nn_r = tmp["nn_l"].tolist()

        if "T" in type:
            left = list(zip(seq_l, count_l, nn_l))
            newleft = [(s, c, nn) for (s, c, nn) in left if nn == 1]
            seq_l, count_l, nn_l = zip(*newleft)

            right = list(zip(seq_r, count_r, nn_r))
            newright = [(s, c, nn) for (s, c, nn) in right if nn == 1]
            seq_r, count_r, nn_r = zip(*newright)

         # type is arbitrary designator to load the data as we feel necessary
        if 'H' in type: # marco adjusted w/ hamming distances to remove those freeloader strands
            if 'C' in type:
                if 'L' in type:
                    data = pd.DataFrame({'sequence': seq_l, 'affinity': count_l})
                if 'R' in type:
                    data = pd.DataFrame({'sequence': seq_r, 'affinity': count_r})
                if 'B20' in type:
                    data = pd.DataFrame({'sequence': seq_l + seq_r, 'affinity': count_l + count_r})
                elif 'B' in type:
                    data = pd.DataFrame({'sequence': seqs, 'affinity': affs})

    data.drop_duplicates(inplace=True, keep='first')
    data.reset_index(drop=True)

    train, validate = train_test_split(data, random_state=seed)
    _train = train.copy()
    _validate = validate.copy()

    binders = (_train["affinity"] > countcutoff)
    _binders = _train[binders].sample(n=binders.sum(), replace=True, random_state=seed)

    nbinders = _train[~binders].sample(n=binders.sum()*n_copies, replace=True, random_state=seed)

    above = sum(binders)
    total = len(binders)
    below = total - above

    weights = [1.0 for x in range(outputnum)]

    if(outputnum == 2):
        weights = [1- below/total, 1- above/total]


    if 'G' in type:
        gen_seqs = pd.read_csv(rel_path+'fake_sequences.csv')
        nbinders = gen_seqs.sample(n=binders.sum()*n_copies, replace=False, random_state=seed)

    _train = pd.concat([_binders, nbinders]).reset_index(drop=True)

    # Fake sequence generation
    # nucs = ["A", "C", "G", "T"]
    # if 'G' in type:
    #     generated_seqs = []
    #     for x in range(100000):
    #         nseq = ''.join([random.choice(nucs) for x in range(20)])
    #         while nseq in train['sequence'] or nseq in generated_seqs or nseq in _validate['sequence']:
    #             nseq = ''.join([random.choice(nucs) for x in range(20)])
    #         generated_seqs.append(nseq)
    #     labels = [0 for x in generated_seqs]
    #     nbinders = pd.DataFrame(list(zip(generated_seqs, labels)), columns=['sequence', 'affinity'])
    #     nbinders.to_csv("./data/fake_sequences.csv")


    binders = (_validate["affinity"] > countcutoff)
    _binders = _validate[~binders].sample(n=n_copies * binders.sum(), replace=True, random_state=seed)

    if 'G' in type:
        _binders = gen_seqs.sample(n=binders.sum() * n_copies, replace=False, random_state=seed)

    _validate = pd.concat([_binders, _validate[binders]]).reset_index(drop=True)


    #     generated_seqs = []
    #     for x in range(len(train[~binders])):
    #         nseq = ''.join([random.choice(nucs) for x in range(20)])
    #         while nseq in train['sequence'] or nseq in generated_seqs or nseq in _validate['sequence']:
    #             nseq = ''.join([random.choice(nucs) for x in range(20)])
    #         generated_seqs.append(nseq)
    #     labels = [0 for x in generated_seqs]
    #     nbinders = pd.DataFrame(list(zip(generated_seqs, labels)), columns=['sequence', 'affinity'])
    #     _validate = pd.concat([_binders, nbinders]).reset_index(drop=True)

    return _train, _validate, weights

# overview of the data
# 1) affinity counts
# 2) Plot affinity vs. # of sequences
def data_report(data):
    Q = data["affinity"].value_counts().to_dict()
    Q = {x: Q[x] for x in sorted(Q)}
    plt.yscale("log")
    plt.plot(list(Q.keys()), list(Q.values()))

# binary labeling
def bi_label(x):
    if x == 1:
        return 0
    elif x >= 2:
        return 1

def tri_label(x):
    if x == 1:
        return 0
    elif 2 <= x < 5:
        return 1
    elif x >= 5:
        return 2

#Inferred count labeling
def bi_labelM(x):
    if x < 10:
        return 0
    elif x >= 10:
        return 1

def tri_labelM(x):
    if x < 500:
        return 0
    elif 500 <= x < 5000:
        return 1
    elif x >= 5000:
        return 2




class NAReader(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, dataset, max_length=20, base_to_id=None, shuffle=True):

        self.dataset = dataset.reset_index(drop=True).drop_duplicates(["sequence"])

        self.shuffle = shuffle
        self.on_epoch_end()

        if base_to_id is None:
            self.base_to_id = {
                "A": 0,
                "C": 1,
                "G": 2,
                "T": 3,
                "U": 3,
                "-": 4
            }
        else:
            self.base_to_id = base_to_id

        self.n_bases = 4
        self.max_length = max_length

        self.train_labels = self.dataset.binary.to_numpy()
        self.train_data = self.dataset.sequence.to_numpy()

    def __getitem__(self, index):

        self.count += 1
        if (self.count % self.dataset.shape[0] == 0):
            self.on_epoch_end()

        seq = self.train_data[index]
        Seq = self.one_hot(seq)
        affinity = self.train_labels[index]
        return seq, Seq, affinity

    def one_hot(self, seq):
        one_hot_vector = np.zeros((self.max_length, self.n_bases), dtype=np.float32)
        for n, base in enumerate(seq):
            one_hot_vector[n][self.base_to_id[base]] = 1
        return one_hot_vector.reshape((1, 1, self.n_bases, self.max_length))

    def __len__(self):
        return self.train_data.shape[0]

    def on_epoch_end(self):
        self.count = 0
        if self.shuffle:
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

class NAContrast(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, dataset, n=1, max_length=20, base_to_id=None, shuffle=True):

        self.dataset = dataset.reset_index(drop=True).drop_duplicates(["sequence"])

        self.shuffle = shuffle
        self.n = n
        self.on_epoch_end()

        if base_to_id is None:
            self.base_to_id = {
                "A": 0,
                "C": 1,
                "G": 2,
                "T": 3,
                "U": 3,
                "-": 4
            }
        else:
            self.base_to_id = base_to_id

        self.n_bases = len(self.base_to_id)
        self.max_length = max_length

        self.train_labels = self.dataset.binary.to_numpy()
        self.train_data = self.dataset.sequence.to_numpy()

        self.labels = list(set(list(self.train_labels)))

    def __getitem__(self, index):

        # self.count += 1
        # if (self.count % self.dataset.shape[0] == 0):
        #    self.on_epoch_end()

        label1 = random.choice(self.labels)
        label2 = random.choice(self.labels)
        while label1 == label2:
            label2 = random.choice(self.labels)

        c1 = (self.dataset.binary == label1)
        c2 = (self.dataset.binary == label2)

        sel1 = self.dataset[c1].sample(n=self.n, replace=True)
        sel2 = self.dataset[c2].sample(n=self.n, replace=True)
        sel = pd.concat([sel1, sel2])

        seq = list(sel.sequence.to_numpy())
        Seq = sel.sequence.apply(self.one_hot).to_numpy()
        one_hot = np.vstack(Seq)
        label = np.vstack(sel.binary.to_numpy())

        return (seq, one_hot, label)

    def one_hot(self, seq):
        one_hot_vector = np.zeros((self.max_length, self.n_bases), dtype=np.float32)
        for n, base in enumerate(seq):
            one_hot_vector[n][self.base_to_id[base]] = 1
        return one_hot_vector.reshape((1, 1, self.n_bases, self.max_length))

    def __len__(self):
        return self.train_data.shape[0]

    def on_epoch_end(self):
        self.count = 0
        if self.shuffle:
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

def my_collate(batch):
    seqs = np.vstack([item[0] for item in batch])
    one_hots = torch.from_numpy(np.vstack([item[1] for item in batch]))
    labels = torch.from_numpy(np.vstack([item[2] for item in batch])).squeeze(1)  # torch.LongTensor(target)
    return [seqs, one_hots, labels]

def one_hot(seq, n_bases=4, max_length=20):
    base_to_id = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "U": 3,
        "-": 4
    }
    one_hot_vector = np.zeros((max_length, n_bases), dtype=np.float32)
    for n, base in enumerate(seq):
        one_hot_vector[n][base_to_id[base]] = 1
    return one_hot_vector.reshape((1, n_bases, max_length))

def one_hot_apply(seq, n_bases=4, max_length=20):
    base_to_id = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "U": 3,
        "-": 4
    }
    one_hot_vector = np.zeros((max_length, n_bases), dtype=np.float32)
    for n, base in enumerate(seq):
        one_hot_vector[n][base_to_id[base]] = 1
    return one_hot_vector.reshape((1, max_length, n_bases))


# adjust nonbinding to binding
test_set_corr = {
    "AGTGATGATGTGTGGTAGGC": 0, # an1
    "AGTGTAGGTGTGGATGATGC": 0, # an2
    "TAGGTTTTGGGTAGCGTGGT": 0, # an3
    "AGGGATGATGTGTGGCAGGA": 0, # an4
    "CTAGGACGGGTAGGGCGGTG": 0, # an5
    "AGGGATGTGTGTGGTAGGCT": 0, # an6
    "AGGGATGCTGCGTGGTAGGC": 1, # an7
    "GAGGGTTGGTGTGGTTGGCA": 1, # an8
    "AGGGTTGGTGTGTGGTTGGC": 1, # an9
    "ATGGTTGGTTTATGGTTGGC": 1, # an10
    "GAAGGGTGGTCAGGGTGGGA": 1, # an11
    "GGAGGGTGGGTCGGGTGGGA": 1, # an12
    "GGGGTTGGTACAGGGTTGGC": 1, # an13
    "AGATGGGCAGGTTGGTGCGG": 1, # an14
    "AGATGGGTGGGTAGGGTGGG": 1, # an15
    "ATAGGGTGGGTGGGTGGGTA": 1, # an16
    "TGGTGGTTGGGTTGGGTTGG": 1, # an17
    "TGGGATGGGATTGGTAGGCG": 0, # an18
    "AGGGTTGGTTATGTGGTTGG": 1, # an19
    "ATTGGTTGGGTAGGGTGGTT": 1, # an20
    "AAACGGTTGGTGAGGTTGGT": 1, # an21
    "CGGGGTGGTGTGGGTGGGAG": 1, # an22
    "TATTGGTTGGATAGGTTGGT": 1, # an23
    "AGGGTTGGGTGGTTGGATGA": 1, # an24
    "CGGGTTGGGGGGTTGGATTC": 1, # an25
    "CGGTTGGGGGGGTTGGATAC": 1, # an26
    "TGTGGGTTGGTGAGGTAGGT": 1, # an27
}

dca_test_set_corr = {
    "AGGGTAGGTGTGGGGTATGC": 0, # a1
    "AGGGTAGATGTGTAGGATGC": 0,# a3
    "AGGGATGATGGTTGGTAGGC": 0, #a4
    "AGGGATGATGTGGATTAGGC": 0, #a5
    "AGGGTGGGAGCGGGGGACGC": 0, # a6
    "CGGGTAGGTGTGGATTATGC": 0, # a7
    "GTAGGACGGGTAGGGCGGTC": 0, # a8
    "GGGGGTTGGGCGGGATGGGC": 0, # a9
    "GCGGGTTGGGCAGGATCAGC": 0, # a10
    "GTAGGATGGGTGGGGTGGGA": 1, # c2
    "GTAGGATGGGTAGGGTGGTA": 1, # c4
    "CTAGGTTGGGTAGGGTGGTG": 1, # c6
    "CTAGCATGGGTAGGGTGGTG": 1, # c7
    "GTAGCATGGGTAGGGTGGTC": 0, # c8
    "TTGGGTGGTGTAGGTTGGCG": 1, # c9
    "TTGGGTGGTGCAGGTTCGCG": 0 # c10
}



class ResNet(nn.Module):
    def __init__(self, fcl_layers=[], dr=0.0, output_size=1, resnet_model=18, pretrained=True, batch_size=64, data_length=20, lr=0.001, weight_decay=0.):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.resnet_model = resnet_model
        if self.resnet_model == 18:
            resnet = models.resnet18(pretrained=self.pretrained)
        elif self.resnet_model == 34:
            resnet = models.resnet34(pretrained=self.pretrained)
        elif self.resnet_model == 50:
            resnet = models.resnet50(pretrained=self.pretrained)
        elif self.resnet_model == 101:
            resnet = models.resnet101(pretrained=self.pretrained)
        elif self.resnet_model == 152:
            resnet = models.resnet152(pretrained=self.pretrained)
        resnet.conv1 = torch.nn.Conv1d(batch_size, data_length, 5, 2, 2, bias=False)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet_output_dim = resnet.fc.in_features
        self.resnet = nn.Sequential(*modules)
        self.fcn = self.make_fcn(self.resnet_output_dim, output_size, fcl_layers, dr)

    def make_fcn(self, input_size, output_size, fcl_layers, dr):
        if len(fcl_layers) > 0:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, fcl_layers[0]),
                nn.BatchNorm1d(fcl_layers[0]),
                torch.nn.LeakyReLU()
            ]
            if len(fcl_layers) == 1:
                fcn.append(nn.Linear(fcl_layers[0], output_size))
            else:
                for i in range(len(fcl_layers) - 1):
                    fcn += [
                        nn.Linear(fcl_layers[i], fcl_layers[i + 1]),
                        nn.BatchNorm1d(fcl_layers[i + 1]),
                        torch.nn.LeakyReLU(),
                        nn.Dropout(dr)
                    ]
                fcn.append(nn.Linear(fcl_layers[i + 1], output_size))
        else:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, output_size)
            ]
        if output_size > 1:
            fcn.append(torch.nn.LogSoftmax(dim=1))
        return nn.Sequential(*fcn)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fcn(x)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):

        super(SpectralNorm, self).__init__()

        self.module = module
        self.name = name
        self.power_iterations = power_iterations

        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Self_Attention(nn.Module):

    """ Self attention Layer
        Based on https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    """

    def __init__(self, in_dim):

        super(Self_Attention, self).__init__()

        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv = SpectralNorm(self.query_conv)
        self.key_conv = SpectralNorm(self.key_conv)
        self.value_conv = SpectralNorm(self.value_conv)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        B, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            B, -1, width*height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(B, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(B, -1, width*height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, width, height)
        out = self.gamma * out + x

        return out, attention

class ATTENTION_VAE(nn.Module):

    def __init__(self,
                 pretrained=True,
                 resnet_model=152,
                 image_channels=1,
                 hidden_dims=[8, 16, 32, 64, 128],
                 z_dim=10,
                 out_image_channels=1,
                 output_size=4,
                 fcl_layers=[],
                 dr=0.0,
                 weights=False):

        super(ATTENTION_VAE, self).__init__()

        self.image_channels = image_channels
        self.hidden_dims = hidden_dims
        self.z_dim = z_dim
        self.out_image_channels = out_image_channels

        self.encoder = None
        self.decoder = None
        self.weights = weights

        self.pretrained = pretrained
        self.resnet_model = resnet_model
        if self.resnet_model == 18:
            resnet = models.resnet18(pretrained=self.pretrained)
        elif self.resnet_model == 34:
            resnet = models.resnet34(pretrained=self.pretrained)
        elif self.resnet_model == 50:
            resnet = models.resnet50(pretrained=self.pretrained)
        elif self.resnet_model == 101:
            resnet = models.resnet101(pretrained=self.pretrained)
        elif self.resnet_model == 152:
            resnet = models.resnet152(pretrained=self.pretrained)
        resnet.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet_output_dim = resnet.fc.in_features
        self.resnet = nn.Sequential(*modules)

        self.encoder_block1 = self.encoder_block(
            self.image_channels, self.hidden_dims[0], 1, 4, 1)
        self.encoder_atten1 = Self_Attention(self.hidden_dims[0])
        self.encoder_block2 = self.encoder_block(
            self.hidden_dims[0], self.hidden_dims[1], 1, 4, 1)
        self.encoder_atten2 = Self_Attention(self.hidden_dims[1])
        self.encoder_block3 = self.encoder_block(
            self.hidden_dims[1], self.hidden_dims[2], 1, 4, 1)
        self.encoder_atten3 = Self_Attention(self.hidden_dims[2])
        self.encoder_block4 = self.encoder_block(
            self.hidden_dims[2], self.hidden_dims[3], 1, 4, 1)
        self.encoder_atten4 = Self_Attention(self.hidden_dims[3])
        self.encoder_block5 = self.encoder_block(
            self.hidden_dims[3], self.hidden_dims[4], 1, 4, 1)
        self.encoder_atten5 = Self_Attention(self.hidden_dims[4])
        #         self.encoder_block6 = self.encoder_block(
        #             self.hidden_dims[4], self.hidden_dims[5], 5, 5, 0)

        self.fc1 = nn.Linear(self.resnet_output_dim, self.z_dim)
        self.fc2 = nn.Linear(self.resnet_output_dim, self.z_dim)

        # Add extra output channel if we are using Matt's physical constraint
        if self.out_image_channels > 1:
            self.hidden_dims = [
                self.out_image_channels * x for x in self.hidden_dims
            ]

        self.fc3 = nn.Linear(self.z_dim, self.hidden_dims[-1])
        self.fcn = self.make_fcn(self.z_dim, output_size, fcl_layers, dr)

        #         self.decoder_block1 = self.decoder_block(
        #             self.hidden_dims[5], self.hidden_dims[4], 1, 4, 1)
        #         self.decoder_atten1 = Self_Attention(self.hidden_dims[4])
        self.decoder_block2 = self.decoder_block(
            self.hidden_dims[4], self.hidden_dims[3], (4, 5), (1, 5), 0)
        self.decoder_atten2 = Self_Attention(self.hidden_dims[3])
        self.decoder_block3 = self.decoder_block(
            self.hidden_dims[3], self.hidden_dims[2], (1, 5), (1, 4), 0)  # (1,5), (1,5)
        self.decoder_atten3 = Self_Attention(self.hidden_dims[2])
        self.decoder_block4 = self.decoder_block(
            self.hidden_dims[2], self.hidden_dims[1], (1, 2), (1, 2), 0)
        self.decoder_atten4 = Self_Attention(self.hidden_dims[1])
        self.decoder_block5 = self.decoder_block(
            self.hidden_dims[1], self.hidden_dims[0], (1, 2), (1, 2), 0)
        self.decoder_atten5 = Self_Attention(self.hidden_dims[0])
        self.decoder_block6 = self.decoder_block(
            self.hidden_dims[0], self.out_image_channels, (1, 2), (1, 2), 0, sigmoid=True)

        # self.load_weights()

    def encoder_block(self, dim1, dim2, kernel_size, stride, padding):
        return nn.Sequential(
            SpectralNorm(
                nn.Conv2d(dim1, dim2, kernel_size=kernel_size,
                          stride=stride, padding=padding)
            ),
            nn.BatchNorm2d(dim2),
            nn.LeakyReLU()
        )

    def decoder_block(self, dim1, dim2, kernel_size, stride, padding, sigmoid=False):
        return nn.Sequential(
            SpectralNorm(
                nn.ConvTranspose2d(
                    dim1, dim2, kernel_size=kernel_size, stride=stride, padding=padding)
            ),
            nn.BatchNorm2d(dim2),
            nn.LeakyReLU() if not sigmoid else nn.Sigmoid()
        )

    def make_fcn(self, input_size, output_size, fcl_layers, dr):
        if len(fcl_layers) > 0:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, fcl_layers[0]),
                nn.BatchNorm1d(fcl_layers[0]),
                torch.nn.LeakyReLU()
            ]
            if len(fcl_layers) == 1:
                fcn.append(nn.Linear(fcl_layers[0], output_size))
            else:
                for i in range(len(fcl_layers) - 1):
                    fcn += [
                        nn.Linear(fcl_layers[i], fcl_layers[i + 1]),
                        nn.BatchNorm1d(fcl_layers[i + 1]),
                        torch.nn.LeakyReLU(),
                        nn.Dropout(dr)
                    ]
                fcn.append(nn.Linear(fcl_layers[i + 1], output_size))
        else:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, output_size)
            ]
        if output_size > 1:
            fcn.append(torch.nn.LogSoftmax(dim=1))
        return nn.Sequential(*fcn)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(std.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        #         h = self.encoder_block1(x)
        #         h, att_map1 = self.encoder_atten1(h)
        #         h = self.encoder_block2(h)
        #         h, att_map2 = self.encoder_atten2(h)
        #         h = self.encoder_block3(h)
        #         h, att_map3 = self.encoder_atten3(h)
        #         h = self.encoder_block4(h)
        #         h, att_map4 = self.encoder_atten4(h)
        #         h = self.encoder_block5(h)
        #         h, att_map5 = self.encoder_atten5(h)

        h = self.resnet(x)
        h = h.view(h.size(0), -1)  # flatten
        z, mu, logvar = self.bottleneck(h)
        return h, z, mu, logvar, None  # [att_map3, att_map4, att_map5]

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), self.hidden_dims[-1], 1, 1)  # flatten/reshape
        z = self.decoder_block2(z)
        z, att_map2 = self.decoder_atten2(z)
        z = self.decoder_block3(z)
        # z, att_map3 = self.decoder_atten3(z)
        # z = self.decoder_block4(z)
        # z, att_map4 = self.decoder_atten4(z)
        # z = self.decoder_block5(z)
        # z, att_map5 = self.decoder_atten5(z)
        # z = self.decoder_block6(z)
        return z, [att_map2]  # , att_map3], att_map4], att_map5]

    def forward(self, x):
        h, z, mu, logvar, encoder_att = self.encode(x)
        out = self.fcn(z)
        z, decoder_att = self.decode(z)
        return out, z, mu, logvar

    def load_weights(self):
        # Load weights if supplied
        if os.path.isfile(self.weights):
            # Load the pretrained weights
            model_dict = torch.load(
                self.weights,
                map_location=lambda storage, loc: storage
            )
            self.load_state_dict(model_dict["model_state_dict"])
            return

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return


class SymmetricMSE:

    def __init__(self, alpha, gamma, kld_weight=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.kld_weight = kld_weight

    def __call__(self, recon_x, x, mu, logvar):
        criterion = nn.MSELoss(reduction='mean')
        #criterion = nn.MSELoss()
        BCE = criterion(recon_x, x)
        # print('recon_x', recon_x.size())
        # print('x', x.size())
        # # print('BCE', BCE.size())
        # print('bce',BCE)
        # KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        #KLD /= logvar.size()[1]
        norm = self.alpha + self.kld_weight * self.gamma

        KLD = torch.clamp_max(KLD, 1000)
        BCE = torch.clamp_max(BCE, 1000)

        # if KLD == np.nan or BCE == np.nan:
        #     print("NAN DETECTED")
        # ret = (self.alpha * BCE + self.kld_weight * self.gamma * KLD) / norm
        return (self.alpha * BCE + self.kld_weight * self.gamma * KLD) / norm, BCE, KLD

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class CosineAnnealingLR_with_Restart(_LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. The original pytorch
    implementation only implements the cosine annealing part of SGDR,
    I added my own implementation of the restarts part.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        T_mult (float): Increase T_max by a factor of T_mult
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        model (pytorch model): The model to save.
        out_dir (str): Directory to save snapshots
        take_snapshot (bool): Whether to save snapshots at every restart
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_mult, model, out_dir, take_snapshot, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.Te = self.T_max
        self.eta_min = eta_min
        self.current_epoch = last_epoch

        self.model = model
        self.out_dir = out_dir
        self.take_snapshot = take_snapshot

        self.lr_history = []

        super(CosineAnnealingLR_with_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lrs = [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.current_epoch / self.Te)) / 2 for base_lr in self.base_lrs]

        self.lr_history.append(new_lrs)
        return new_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        ## restart
        if self.current_epoch == self.Te:
            #print("restart at epoch {:.5f}".format(self.last_epoch + 1))

            #if self.take_snapshot:
            #    torch.save(
            #        {"epoch": self.T_max, "state_dict": self.model.state_dict()}, self.out_dir + "/" + "snapshot_e_{:.5f}.pth.tar".format(self.T_max)
            #    )

            ## reset epochs since the last reset
            self.current_epoch = 0

            ## reset the next goal
            self.Te = int(self.Te * self.T_mult)
            self.T_max = self.T_max + self.Te

    class SmoothCrossEntropyLoss(_WeightedLoss):
        def __init__(self, weight=None, reduction='mean', smoothing=0.0):
            super().__init__(weight=weight, reduction=reduction)
            self.smoothing = smoothing
            self.weight = weight
            self.reduction = reduction

        def k_one_hot(self, targets: torch.Tensor, n_classes: int, smoothing=0.0):
            with torch.no_grad():
                targets = torch.empty(size=(targets.size(0), n_classes),
                                      device=targets.device) \
                    .fill_(smoothing / (n_classes - 1)) \
                    .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
            return targets

        def reduce_loss(self, loss):
            return loss.mean() if self.reduction == 'mean' else loss.sum() \
                if self.reduction == 'sum' else loss

        def forward(self, inputs, targets):
            assert 0 <= self.smoothing < 1

            targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
            log_preds = F.log_softmax(inputs, -1)

            if self.weight is not None:
                log_preds = log_preds * self.weight.unsqueeze(0)

            return self.reduce_loss(-(targets * log_preds).sum(dim=-1))


class drelu(nn.Module):
    '''
    Activation Function Taken from Tubiana et. al
    '''
    def __init__(self, in_features, tp=None, tm=None, ap=None, am=None):
        super().__init__()
        if tp is None:
            self.theta_plus = Parameter(torch.zeros([in_features]), requires_grad=True)
        else:
            self.theta_plus = Parameter(tp, requires_grad=True)

        if tm is None:
            self.theta_minus = Parameter(torch.zeros([in_features]), requires_grad=True)
        else:
            self.theta_minus = Parameter(tm, requires_grad=True)

        if ap is None:
            self.a_plus = Parameter(torch.ones([in_features]), requires_grad=True)
        else:
            self.a_plus = Parameter(ap, requires_grad=True)

        if am is None:
            self.a_minus = Parameter(torch.ones([in_features]), requires_grad=True)
        else:
            self.a_minus = Parameter(am, requires_grad=True)

    def forward(self, input): # Interpreted as H(I) on input from visble layers
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """

        zero = torch.zeros_like(input, dtype=torch.float)
        pos = torch.max(input, zero)
        neg = torch.min(input, zero)

        return self.a_plus * torch.pow(pos, 2) + 0.5 * self.a_minus * torch.pow(neg, 2) + self.theta_plus*pos + self.theta_minus*neg
