import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
import argparse
import json
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import sklearn
from ray import tune
import numpy as np
import seaborn
import matplotlib.pyplot as plt

import os
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchvision import transforms
import MLmodels as m
from torch.utils.data import Dataset
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.suggest.bayesopt import BayesOptSearch

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    From: https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

class NAReader_Siamese(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """
    def __init__(self, dataset, max_length=20, base_to_id=None, shuffle=True):

        self.dataset = dataset.reset_index(drop=True).drop_duplicates(["sequence"])

        binders = (dataset["binary"] == 1)

        seq1, seq2, label = [], [], []
        for index, row in self.dataset[binders].iterrows():
            # good example
            s1 = row['sequence']
            s2 = str(self.dataset[binders].sample(n=1, replace=True).iloc[0]['sequence'])
            # print(s1, s2)
            while s1 == s2:
                s2 = str(self.dataset[binders].sample(n=1, replace=True).iloc[0]['sequence'])

            seq1.append(s1)
            seq2.append(s2)
            label.append(1)
            # bad example
            b2 = str(self.dataset[~binders].sample(n=1, replace=True).iloc[0]['sequence'])
            seq1.append(s1)
            seq2.append(b2)
            label.append(0)

        self.dataset = pd.DataFrame({'sequence1':seq1, 'sequence2':seq2, 'label':label})

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

        self.train_labels = self.dataset.label.to_numpy()
        self.train_data1 = self.dataset.sequence1.to_numpy()
        self.train_data2 = self.dataset.sequence2.to_numpy()

    def __getitem__(self, index):

        self.count += 1
        if (self.count % self.dataset.shape[0] == 0):
            self.on_epoch_end()

        seq1 = self.train_data1[index]
        seq2 = self.train_data2[index]
        ohe1 = self.one_hot(seq1)
        ohe2 = self.one_hot(seq2)
        label = self.train_labels[index]
        return seq1, seq2, ohe1, ohe2, label

    def one_hot(self, seq):
        one_hot_vector = np.zeros((self.max_length, self.n_bases), dtype=np.float32)
        for n, base in enumerate(seq):
            one_hot_vector[n][self.base_to_id[base]] = 1
        return one_hot_vector.reshape((1, 1, self.n_bases, self.max_length))

    def __len__(self):
        return self.train_data1.shape[0]

    def on_epoch_end(self):
        self.count = 0
        if self.shuffle:
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

class NAReader_Siamese_Verification(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """
    def __init__(self, dataset, max_length=20, base_to_id=None, shuffle=True):

        self.dataset = dataset.reset_index(drop=True).drop_duplicates(["sequence"])

        binders = (dataset["binary"] == 1)

        seq1, seq2, label = [], [], []
        for index, row in self.dataset[binders].iterrows():
            # good example
            s1 = row['sequence']
            for ind2, row2 in self.dataset[binders].iterrows():
                if index >= ind2:
                    continue
                else:
                    s2 = row2['sequence']
                    seq1.append(s1)
                    seq2.append(s2)
                    label.append(1)

        for index, row in self.dataset[binders].iterrows():
            # good example
            s1 = row['sequence']
            for ind2, row2 in self.dataset[~binders].iterrows():
                # if index >= ind2:
                #     continue
                # else:
                s2 = row2['sequence']
                seq1.append(s1)
                seq2.append(s2)
                label.append(0)

        print(f"Siamese Verification Positive Ex. Num: {label.count(1)}, Negative Ex. Num {label.count(0)}")

        self.dataset = pd.DataFrame({'sequence1':seq1, 'sequence2':seq2, 'label':label})

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

        self.train_labels = self.dataset.label.to_numpy()
        self.train_data1 = self.dataset.sequence1.to_numpy()
        self.train_data2 = self.dataset.sequence2.to_numpy()

    def __getitem__(self, index):

        self.count += 1
        if (self.count % self.dataset.shape[0] == 0):
            self.on_epoch_end()

        seq1 = self.train_data1[index]
        seq2 = self.train_data2[index]
        ohe1 = self.one_hot(seq1)
        ohe2 = self.one_hot(seq2)
        label = self.train_labels[index]
        return seq1, seq2, ohe1, ohe2, label

    def one_hot(self, seq):
        one_hot_vector = np.zeros((self.max_length, self.n_bases), dtype=np.float32)
        for n, base in enumerate(seq):
            one_hot_vector[n][self.base_to_id[base]] = 1
        return one_hot_vector.reshape((1, 1, self.n_bases, self.max_length))

    def __len__(self):
        return self.train_data1.shape[0]

    def on_epoch_end(self):
        self.count = 0
        if self.shuffle:
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

def my_collate(batch):
    seqs1 = np.vstack([item[0] for item in batch])
    seqs2 = np.vstack([item[1] for item in batch])
    one_hots1 = torch.from_numpy(np.vstack([item[2] for item in batch]))
    one_hots2 = torch.from_numpy(np.vstack([item[3] for item in batch]))
    labels = torch.from_numpy(np.vstack([item[4] for item in batch])).squeeze(1)
    return [seqs1, seqs2, one_hots1, one_hots2, labels]

class SiameseClassifier(pl.LightningModule):

    def __init__(self, config, outputs=2):
        super(SiameseClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(2, 3), padding=(1, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), padding=(1, 1)),
            nn.Conv2d(64, 128, kernel_size=(2, 3), padding=(1 ,2), stride=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=(2, 3), padding=(1, 2), stride=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d((2,4)),
            nn.Conv2d(128, 256, (1,1)),
            nn.ReLU(),
        )

        self.liner = nn.Sequential(nn.Linear(512, 256), nn.Sigmoid())
        self.out = nn.Linear(256, outputs)

        self.lr = config['lr']
        self.dr = config['dr']
        self.distance_cutoff = config['distance_cutoff'] # Distance for how far examples can be to be considered (1 the same) < distance_cutoff or (0 different) > distance_cutoff
        self.datatype = config['datatype']
        self.batch_size = config['batch_size']

        if 'B' in self.datatype and '20' not in self.datatype:
            self.data_length = 40
        else:
            self.data_length = 20

        self.criterion = ContrastiveLoss(margin=10)
        self.optimizer = SGD
        self.training_data = None
        self.validation_data = None
        self.val_acc_tmp = []
        self.train_acc_tmp = []
        self.val_dists = []

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.dr)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2


    def prepare_data(self):
        # import our data
        train, validate, weights = m.get_rawdata(self.datatype, 10, 5, round=8)
        _train = train.copy()
        _validate = validate.copy()

        # Assigns labels for learning
        _train["binary"] = _train["affinity"].apply(m.bi_labelM)
        _validate["binary"] = _validate["affinity"].apply(m.bi_labelM)

        self.training_data = _train
        self.validation_data = _validate


    def train_dataloader(self):
        # Data Loading
        train_reader = NAReader_Siamese(self.training_data, shuffle=True, max_length=self.data_length)

        train_loader = torch.utils.data.DataLoader(
            train_reader,
            batch_size=self.batch_size,
            # batch_size=self.batch_size,
            collate_fn=my_collate,
            num_workers=4,
            # pin_memory=True,
            shuffle=True
        )

        return train_loader

    def val_dataloader(self):
        # Data Loading
        val_reader = NAReader_Siamese(self.validation_data, shuffle=False, max_length=self.data_length)

        val_loader = torch.utils.data.DataLoader(
            val_reader,
            batch_size=self.batch_size,
            collate_fn=my_collate,
            num_workers=4,
            # pin_memory=True,
            shuffle=False
        )

        return val_loader

    def training_step(self, batch, batch_idx):
        seq1, seq2, x1, x2, y = batch
        out1, out2 = self(x1, x2)
        train_loss = self.criterion(out1, out2, y)
        euclidean_distance = F.pairwise_distance(out1, out2)

        tmp = 1. if euclidean_distance.cpu().detach().numpy() < self.distance_cutoff else 0

        train_acc = 1. if tmp == y.cpu().detach().numpy() else 0.  # we'll try this

        # perform logging
        self.log("ptl/train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_accuracy", train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        seq1, seq2, x1, x2, y = batch
        out1, out2 = self(x1, x2)
        val_loss = self.criterion(out1, out2, y)
        euclidean_distance = F.pairwise_distance(out1, out2)

        tmp = 1. if euclidean_distance.cpu().detach().numpy() < self.distance_cutoff else 0

        val_acc = 1. if tmp == y.cpu().detach().numpy() else 0.   # we'll try this

        self.log("ptl/val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_accuracy", val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": val_loss, "val_acc": val_acc}


######################## Ray Tune Hyperparameter Section #########################

def train_siamese(config, checkpoint_dir=None, num_epochs=10, num_gpus=0):
    trainer = pl.Trainer(
        # default_root_dir="./checkpoints/",
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "loss": "ptl/val_loss",
                    "acc": "ptl/val_accuracy"
                },
                filename="checkpoint",
                on="validation_end")
        ]
    )

    if checkpoint_dir:
        # Workaround:
        ckpt = pl_load(
            os.path.join(checkpoint_dir, "checkpoint"),
            map_location=lambda storage, loc: storage)
        model = SiameseClassifier._load_model_state(
            ckpt, config=config)
        trainer.current_epoch = ckpt["epoch"]
    else:
        model = SiameseClassifier(config, outputs=2)

    trainer.fit(model)

class CustomStopper(tune.Stopper):
    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id, result):
        max_iter = 100
        if not self.should_stop and result["acc"] > 0.96:
            self.should_stop = True
        return self.should_stop or result["training_iteration"] >= max_iter

    def stop_all(self):
        return self.should_stop

def pbt_siamese(datatype, num_samples=10, num_epochs=10, gpus_per_trial=0, cpus_per_trial=1):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": 1,
        "dr": tune.loguniform(0.005, 0.05),
        "distance_cutoff": tune.uniform(0.05, 5.0),
        "datatype": datatype
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        hyperparam_mutations={
            # distribution for resampling
            "lr": lambda: np.random.uniform(0.0001, 0.1),
            "dr": lambda: np.random.uniform(0.005, 0.05),
            "distance_cutoff": lambda: np.random.uniform(0.05, 5.0)
            # allow perturbations within this set of categorical values
            # "momentum": [0.8, 0.9, 0.99],
        })

    reporter = CLIReporter(
        parameter_columns=["lr", "dr", "distance_cutoff"],
        metric_columns=["loss", "acc", "training_iteration"])

    stopper = CustomStopper()

    analysis = tune.run(
        tune.with_parameters(
            train_siamese,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="acc",
        mode="max",
        local_dir="./ray_results/",
        config=config,
        num_samples=num_samples,
        name="tune_pbt_siamese",
        scheduler=scheduler,
        progress_reporter=reporter,
        verbose=1,
        stop=stopper,
        # export_formats=[ExportFormat.MODEL],
        checkpoint_score_attr="acc",
        keep_checkpoints_num=4)

    print("Best hyperparameters found were: ", analysis.best_config)

def exp_results_check(checkpoint_path, result_path, title):
    # example
    # checkpoint_file = './ray_results/tune_vae_asha/train_vae_a45d1_00000_0_batch_size=64,dr=0.029188,lr=0.0075796,z_dim=10_2021-07-13_12-50-57/checkpoints/epoch=28-step=15891.ckpt'
    checkpoint_file = checkpoint_path
    param_file = open(result_path, 'r')
    check_epoch = int(checkpoint_file.split("epoch=", 1)[1].split('-', 1)[0])
    resultjsons = param_file.read().split('\n')
    results = json.loads(resultjsons[check_epoch + 1])
    params = results['config']
    lr = params['lr']
    dr = params['dr']
    batch_size = params['batch_size']
    datatype = params['datatype']
    distance_cutoff = params['distance_cutoff']

    con = {'lr': lr, 'dr': dr, 'batch_size': batch_size, 'datatype': datatype, 'distance_cutoff': distance_cutoff}

    model = SiameseClassifier(con, outputs=2)

    checkpoint = torch.load(checkpoint_file)
    model.prepare_data()

    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    test_set = m.test_set_corr

    test_dict = pd.DataFrame({'sequence': list(test_set.keys()), 'binary': list(test_set.values())})

    # neg_ex = list(test_set.values()).count(0)
    # pos_ex = list(test_set.values()).count(1)

    ver_reader = NAReader_Siamese_Verification(test_dict, shuffle=False)

    ver_loader = torch.utils.data.DataLoader(
        ver_reader,
        batch_size=batch_size,
        collate_fn=my_collate,
        # num_workers=4,
        # pin_memory=True,
        shuffle=False
    )

    tmp_va = []
    preds = []
    truth = []
    for i, batch in enumerate(ver_loader):
        seq1, seq2, x1, x2, y = batch
        out1, out2 = model(x1, x2)

        euclidean_distance = F.pairwise_distance(out1, out2)

        tmp = 1. if euclidean_distance.cpu().detach().numpy() < distance_cutoff else 0

        preds.append(tmp)
        va = 1. if tmp == y.cpu().detach().numpy() else 0.  # we'll try this
        tmp_va.append(va)
        truth.append(bool(y.cpu().detach()))

    # Make confusion Matrix
    y_true = truth
    y_pred = np.asarray(preds, dtype=np.bool).tolist()
    # score = np.asarray([1 if x == y_pred[xid] else 0 for xid, x in enumerate(y_true)])
    # ver_acc = np.mean(score)


    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='true')
    df_cm = pd.DataFrame(cm, index=[0, 1], columns=[0, 1])
    plt.figure(figsize=(10, 7))
    ax = plt.subplot()
    seaborn.set(font_scale=3.0)
    seaborn.heatmap(df_cm, annot=True, ax=ax)
    label_font = {'size': '26'}
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])
    # plt.title(title)
    plt.savefig('../../Dropbox (ASU)/Projects/Aptamer_ML/' + title)


    ver_acc = np.mean(tmp_va)
    o = open('../../Dropbox (ASU)/Projects/Aptamer_ML/' + title + 'results.txt', "w+")
    print("Validation Loss", results['loss'], file=o)
    print("Validation Accuracy", results['acc'], file=o)
    print("Verification Accuracy:", ver_acc, "of dataset size:", len(tmp_va), file=o)
    o.close()

def val_results_check(checkpoint_path, result_path, title):
    # example
    # checkpoint_file = './ray_results/tune_vae_asha/train_vae_a45d1_00000_0_batch_size=64,dr=0.029188,lr=0.0075796,z_dim=10_2021-07-13_12-50-57/checkpoints/epoch=28-step=15891.ckpt'
    checkpoint_file = checkpoint_path
    param_file = open(result_path, 'r')
    check_epoch = int(checkpoint_file.split("epoch=", 1)[1].split('-', 1)[0])
    resultjsons = param_file.read().split('\n')
    results = json.loads(resultjsons[check_epoch + 1])
    params = results['config']
    lr = params['lr']
    dr = params['dr']
    batch_size = params['batch_size']
    datatype = params['datatype']
    distance_cutoff = params['distance_cutoff']
    con = {'lr': lr, 'dr': dr, 'batch_size': batch_size, 'datatype': datatype, 'distance_cutoff': distance_cutoff}
    # model = ResNetClassifier(con, 2, 152, optimizer='adam')
    model = SiameseClassifier(con, outputs=2)

    checkpoint = torch.load(checkpoint_file)
    model.prepare_data()
    # model.criterion.weight = torch.tensor([0., 0.]) # need to add as this is saved by the checkpoint file
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    vd = model.val_dataloader()

    tmp_va = []
    preds = []
    truth = []
    for i, batch in enumerate(vd):
        seq1, seq2, x1, x2, y = batch
        out1, out2 = model(x1, x2)
        # val_loss = self.criterion(out1, out2, y)
        euclidean_distance = F.pairwise_distance(out1, out2)
        # print("distance", euclidean_distance)
        # self.val_dists.append(euclidean_distance)
        tmp = 1. if euclidean_distance.cpu().detach().numpy() < distance_cutoff else 0
        # pred = torch.tensor([1.]) if euclidean_distance < torch.tensor([self.distance_cutoff]) else torch.tensor([0.])
        preds.append(tmp)
        va = 1. if tmp == y.cpu().detach().numpy() else 0.  # we'll try this
        tmp_va.append(va)
        truth.append(bool(y.cpu().detach()))

    # Make confusion Matrix
    y_true = truth
    y_pred = np.asarray(preds, dtype=np.bool).tolist()

    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='true')
    df_cm = pd.DataFrame(cm, index=[0, 1], columns=[0, 1])
    plt.figure(figsize=(10, 7))
    ax = plt.subplot()
    seaborn.set(font_scale=3.0)
    seaborn.heatmap(df_cm, annot=True, ax=ax)
    label_font = {'size': '26'}
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])
    # plt.title(title)
    plt.savefig('../../Dropbox (ASU)/Projects/Aptamer_ML/' + title + "_VER")

    ver_acc = np.mean(tmp)
    o = open('../../Dropbox (ASU)/Projects/Aptamer_ML/' + title + 'results_ver.txt', "w+")
    print("Validation Loss", results['loss'], file=o)
    print("Validation Accuracy", results['acc'], file=o)
    print("Verification Accuracy:", ver_acc, "of dataset size:", len(y_true), file=o)
    o.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resnet Training on Aptamer Dataset")
    parser.add_argument('dataset', type=str, help="3-7 letter/number abbreviation describing subset of the data to use")
    parser.add_argument('cpus_per_trial', type=str, help="Number of cpus available to each trial in Ray Tune")
    parser.add_argument('gpus_per_trial', type=str, help="Number of gpus available to each trial in Ray Tune")
    parser.add_argument('samples', type=str, help="Number of Ray Tune Samples")
    args = parser.parse_args()
    os.environ["SLURM_JOB_NAME"] = "bash"

    ## population based training of Siamese Network
    pbt_siamese(args.dataset, int(args.samples), 30, gpus_per_trial=int(args.gpus_per_trial),
              cpus_per_trial=int(args.cpus_per_trial))

    ## Debugging
    # con = {'lr': 1e-3, 'dr': 0.1, 'batch_size': 1, 'datatype': 'HGCLT', 'distance_cutoff': 2.}
    # model = SiameseClassifier(con, outputs=2)

    ## Single Loop Debugging

    # model.prepare_data()
    # d = model.train_dataloader()
    # for i, batch in enumerate(d):
    #     if i > 0:
    #         break
    #     else:
    #         model.training_step(batch, i)
    #         model.validation_step(batch, i)


    # pytorch lightning loop
    # sn = SiameseClassifier(con, outputs=2)
    # plt = pl.Trainer(max_epochs=1, gpus=1)
    # plt.fit(sn)



