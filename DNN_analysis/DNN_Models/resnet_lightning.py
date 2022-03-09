import pandas as pd
import seaborn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
import argparse
import json
import pytorch_lightning as pl
import pandas as pd
import sklearn
from ray import tune
import numpy as np
import matplotlib.pyplot as plt  # just for confusion matrix generation

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
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.suggest.bayesopt import BayesOptSearch

from hagelslag.evaluation.ProbabilityMetrics import *
from hagelslag.evaluation.MetricPlotter import *


class ResNetClassifier(pl.LightningModule):
    def __init__(self, config, num_classes, resnet_version,
                test_path=None,
                 optimizer='adam',
                 transfer=True):
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        # hyperparameters
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        # for importing different versions of the data
        self.datatype = config['datatype']

        if 'B' in self.datatype and '20' not in self.datatype:
            self.data_length = 40
        else:
            self.data_length = 20

        self.training_data = None
        self.validation_data = None


        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features

        # replace final layer for fine tuning
        fcn = [
            nn.Dropout(config['dr']),
            nn.Linear(linear_size, num_classes)
        ]
        if num_classes > 1:
            fcn.append(torch.nn.LogSoftmax(dim=1))

        self.fcn = nn.Sequential(*fcn)
        self.resnet_model.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)

        modules = list(self.resnet_model.children())[:-1]  # delete the last fc layer.
        self.resnet_model = nn.Sequential(*modules)

    def forward(self, X):
        x = self.resnet_model(X)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fcn(x)
        return x

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def prepare_data(self):
        # import our data
        train, validate, weights = m.get_rawdata(self.datatype, 10, 5, round=8)
        _train = train.copy()
        _validate = validate.copy()

        # Assigns labels for learning
        _train["binary"] = _train["affinity"].apply(m.bi_labelM)
        _validate["binary"] = _validate["affinity"].apply(m.bi_labelM)

        _weights = torch.FloatTensor(weights)
        # instantiate loss criterion, need weights so put this here
        self.criterion = m.SmoothCrossEntropyLoss(weight=_weights, smoothing=0.01)

        self.training_data = _train
        self.validation_data = _validate


    def train_dataloader(self):
        # Data Loading
        train_reader = m.NAReader(self.training_data, shuffle=True, max_length=self.data_length)

        train_loader = torch.utils.data.DataLoader(
            train_reader,
            batch_size=self.batch_size,
            # batch_size=self.batch_size,
            collate_fn=m.my_collate,
            num_workers=4,
            # pin_memory=True,
            shuffle=True
        )

        return train_loader

    def training_step(self, batch, batch_idx):
        seq, x, y = batch
        softmax = self(x)
        train_loss = self.criterion(softmax, y)

        # Convert to labels
        preds = torch.argmax(softmax, 1).clone().double() # convert to torch float 64

        predcpu = list(preds.detach().cpu().numpy())
        ycpu = list(y.detach().cpu().numpy())
        train_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)

        # perform logging
        self.log("ptl/train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_accuracy", train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def val_dataloader(self):
        # Data Loading
        val_reader = m.NAReader(self.validation_data, shuffle=False)

        val_loader = torch.utils.data.DataLoader(
            val_reader,
            batch_size=self.batch_size,
            collate_fn=m.my_collate,
            num_workers=4,
            # pin_memory=True,
            shuffle=False
        )

        return val_loader

    def validation_step(self, batch, batch_idx):
        seq, x, y = batch
        softmax = self(x)
        val_loss = self.criterion(softmax, y)

        # Convert to labels
        preds = torch.argmax(softmax, 1).clone().double()  # convert to torch float 64

        predcpu = list(preds.detach().cpu().numpy())
        ycpu = list(y.detach().cpu().numpy())
        val_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)

        # perform logging
        self.log("ptl/val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_accuracy", val_acc, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": val_loss, "val_acc": val_acc}


def train_resnet(config, checkpoint_dir=None, num_epochs=10, num_gpus=0):
    trainer = pl.Trainer(
        # default_root_dir="./checkpoints/",
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
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
        model = ResNetClassifier._load_model_state(
            ckpt, config=config)
        trainer.current_epoch = ckpt["epoch"]
    else:
        model = ResNetClassifier(config, 2, 152, optimizer='adam')

    trainer.fit(model)

def tune_asha(datatype, num_samples=10, num_epochs=10, gpus_per_trial=0, cpus_per_trial=1):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "dr": tune.loguniform(0.005, 0.5),
        "datatype": datatype
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=5,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size"],
        metric_columns=["loss", "acc", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_resnet,
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
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_res_asha")

    print("Best hyperparameters found were: ", analysis.best_config)
    # analysis.to_csv('~/ray_results/' + config['datatype'])

def tune_asha_search(datatype, num_samples=10, num_epochs=10, gpus_per_trial=0, cpus_per_trial=1):
    config = {
        "lr": tune.uniform(1e-4, 1e-1),
        #"batch_size": tune.choice([32, 64, 128]),
        "dr": tune.uniform(0.005, 0.5),
        "datatype": datatype,
        "batch_size": 32
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=5,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "dr"],
        metric_columns=["loss", "acc", "training_iteration"])

    bayesopt = BayesOptSearch(metric="mean_loss", mode="min")
    analysis = tune.run(
        tune.with_parameters(
            train_resnet,
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
        search_alg=bayesopt,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_res_bayopt")

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

    con = {'lr': lr, 'dr': dr, 'batch_size': batch_size, 'datatype': datatype}
    model = ResNetClassifier(con, 2, 152, optimizer='adam')

    checkpoint = torch.load(checkpoint_file)
    model.prepare_data()
    model.criterion.weight = torch.tensor([0., 0.]) # need to add as this is saved by the checkpoint file
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    test_set = m.test_set_corr
    dca_set = m.dca_test_set_corr
    verdict = {'sequence':list(test_set.keys()), 'binary':list(test_set.values())}
    _verification = pd.DataFrame(verdict)
    ver_reader = m.NAReader(_verification, shuffle=False)

    ver_loader = torch.utils.data.DataLoader(
        ver_reader,
        batch_size=len(test_set.keys()),
        collate_fn=m.my_collate,
        num_workers=1,
        # pin_memory=True,
        shuffle=False
    )

    for i, batch in enumerate(ver_loader):
        seqs, ohe, labels = batch
        softmax = model(ohe)

        preds = torch.argmax(softmax, 1).clone().double()  # convert to torch float 64
        predcpu = list(preds.detach().cpu().numpy())
        ycpu = labels
        # ver_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)

        # Make confusion Matrix
        y_true = ycpu.detach().cpu().numpy().astype('bool').tolist()
        y_pred = np.asarray(predcpu, dtype=np.bool).tolist()
        score = np.asarray([1 if x == y_pred[xid] else 0 for xid, x in enumerate(y_true)])
        ver_acc = np.mean(score)

        f1 = sklearn.metrics.f1_score(y_true, y_pred)
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

    o = open('../../Dropbox (ASU)/Projects/Aptamer_ML/'+title+'results.txt', "w+")
    print("Validation Loss", results['loss'], file=o)
    print("Validation Accuracy", results['acc'], file=o)
    print("Verification Accuracy:", ver_acc, "of dataset size:", len(test_set.keys()), file=o)
    print("F1-score", f1, file=o)
    o.close()

def exp_results_check_progress(checkpoint_path, hparams, progress, title):
    # example
    # checkpoint_file = './ray_results/tune_vae_asha/train_vae_a45d1_00000_0_batch_size=64,dr=0.029188,lr=0.0075796,z_dim=10_2021-07-13_12-50-57/checkpoints/epoch=28-step=15891.ckpt'
    checkpoint_file = checkpoint_path
    # param_file = open(result_path, 'r')
    # check_epoch = int(checkpoint_file.split("epoch=", 1)[1].split('-', 1)[0])
    # resultjsons = param_file.read().split('\n')
    o = open(hparams, 'r')
    params = json.load(o)
    # params = results['config']
    lr = params['lr']
    dr = params['dr']
    batch_size = params['batch_size']
    datatype = params['datatype']

    progress = pd.read_csv(progress)
    loss = progress.iloc[-1].loss
    acc = progress.iloc[-1].acc

    con = {'lr': lr, 'dr': dr, 'batch_size': batch_size, 'datatype': datatype}
    model = ResNetClassifier(con, 2, 152, optimizer='adam')


    checkpoint = torch.load(checkpoint_file)
    model.prepare_data()
    model.criterion.weight = torch.tensor([0., 0.])  # need to add as this is saved by the checkpoint file
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    test_set = m.test_set_corr
    verdict = {'sequence': list(test_set.keys()), 'binary': list(test_set.values())}
    _verification = pd.DataFrame(verdict)
    ver_reader = m.NAReader(_verification, shuffle=False)

    ver_loader = torch.utils.data.DataLoader(
        ver_reader,
        batch_size=len(test_set.keys()),
        collate_fn=m.my_collate,
        # num_workers=4,
        # pin_memory=True,
        shuffle=False
    )

    for i, batch in enumerate(ver_loader):
        seqs, ohe, labels = batch
        softmax = model(ohe)

        preds = torch.argmax(softmax, 1).clone().double()  # convert to torch float 64
        predcpu = list(preds.detach().cpu().numpy())
        ycpu = labels
        # ver_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)

        # Make confusion Matrix
        y_true = ycpu.detach().cpu().numpy().astype('bool').tolist()
        y_pred = np.asarray(predcpu, dtype=np.bool).tolist()
        score = np.asarray([1 if x == y_pred[xid] else 0 for xid, x in enumerate(y_true)])
        ver_acc = np.mean(score)

        f1 = sklearn.metrics.f1_score(y_true, y_pred)

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

    o = open('../../Dropbox (ASU)/Projects/Aptamer_ML/' + title + 'results.txt', "w+")
    print("Validation Loss", loss, file=o)
    print("Validation Accuracy", acc, file=o)
    print("Verification Accuracy:", ver_acc, "of dataset size:", len(test_set.keys()), file=o)
    o.close()


def val_results_check(checkpoint_path, hparams, progress, result_path, title, r=True):
    # example
    # checkpoint_file = './ray_results/tune_vae_asha/train_vae_a45d1_00000_0_batch_size=64,dr=0.029188,lr=0.0075796,z_dim=10_2021-07-13_12-50-57/checkpoints/epoch=28-step=15891.ckpt'
    checkpoint_file = checkpoint_path
    if r:
        param_file = open(result_path, 'r')
        check_epoch = int(checkpoint_file.split("epoch=", 1)[1].split('-', 1)[0])
        resultjsons = param_file.read().split('\n')
        results = json.loads(resultjsons[check_epoch + 1])
        params = results['config']
        lr = params['lr']
        dr = params['dr']
        batch_size = params['batch_size']
        datatype = params['datatype']
        loss = results['loss']
        acc = results['acc']
    else:
        o = open(hparams, 'r')
        params = json.load(o)
        # params = results['config']
        lr = params['lr']
        dr = params['dr']
        batch_size = params['batch_size']
        datatype = params['datatype']

        progress = pd.read_csv(progress)
        loss = progress.iloc[-1].loss
        acc = progress.iloc[-1].acc

    con = {'lr': lr, 'dr': dr, 'batch_size': batch_size, 'datatype': datatype}
    model = ResNetClassifier(con, 2, 152, optimizer='adam')

    checkpoint = torch.load(checkpoint_file)
    model.prepare_data()
    model.criterion.weight = torch.tensor([0., 0.])  # need to add as this is saved by the checkpoint file
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    vd = model.val_dataloader()

    yt, yp = [], []
    for i, batch in enumerate(vd):
        seqs, ohe, labels = batch
        softmax = model(ohe)

        preds = torch.argmax(softmax, 1).clone().double()  # convert to torch float 64
        predcpu = list(preds.detach().cpu().numpy())
        ycpu = labels


        # Make confusion Matrix
        y_true = ycpu.detach().cpu().numpy().astype('bool').tolist()
        y_pred = np.asarray(predcpu, dtype=np.bool).tolist()
        yt += y_true
        yp += y_pred

    # ver_acc = sklearn.metrics.balanced_accuracy_score(yt, yp)
    ver_acc = np.mean(np.asarray([1 if x == yp[xid] else 0 for xid, x in enumerate(yt)]))

    cm = sklearn.metrics.confusion_matrix(yt, yp, normalize='true')

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

    o = open('../../Dropbox (ASU)/Projects/Aptamer_ML/' + title + 'results_ver.txt', "w+")
    print("Validation Loss", loss, file=o)
    print("Validation Accuracy", acc, file=o)
    print("Verification Accuracy:", ver_acc, "of dataset size:", len(yt), file=o)
    o.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resnet Training on Aptamer Dataset")
    parser.add_argument('dataset', type=str, help="3-7 letter/number abbreviation describing subset of the data to use")
    parser.add_argument('cpus_per_trial', type=str, help="Number of cpus available to each trial in Ray Tune")
    parser.add_argument('gpus_per_trial', type=str, help="Number of gpus available to each trial in Ray Tune")
    parser.add_argument('samples', type=str, help="Number of Ray Tune Samples")
    args = parser.parse_args()
    os.environ["SLURM_JOB_NAME"] = "bash"

    tune_asha(args.dataset, int(args.samples), 30, gpus_per_trial=int(args.gpus_per_trial), cpus_per_trial=int(args.cpus_per_trial))
    # tune_asha_search(args.dataset, int(args.samples), 50, gpus_per_trial=int(args.gpus_per_trial), cpus_per_trial=int(args.cpus_per_trial))

    ### Debugging
    # con = {'lr': 1e-3, 'dr': 0.1, 'batch_size': 32, 'datatype': 'HCB20T'}
    #
    # model = ResNetClassifier(con, 2, 18)
    #
    ## Single Loop debugging
    # model.prepare_data()
    # # d = model.train_dataloader()
    # d = model.val_dataloader()
    # print(len(d))
    # total = 0
    # for i, batch in enumerate(d):
    #     if i > 0:
    #         total += 32
    #     else:
    #         total += 32
    #         model.training_step(batch, i)
    #
    # print(total)

    ## pytorch lightning loop
    # rn = ResNetClassifier(con, 2, 152, optimizer='adam')
    # plt = pl.Trainer()
    # plt.fit(rn)















