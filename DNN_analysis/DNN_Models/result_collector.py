import MLmodels as m
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


from resnet_lightning import ResNetClassifier as RC
from resnet_fc_lightning import ResNetClassifier as RCF
from VAE_lightning_v2 import ATTENTION_VAE as VAEL
from VAE_lightning_short import ATTENTION_VAE as VAES
from siamese_lightning import SiameseClassifier as SC
from siamese_lightning import NAReader_Siamese_Verification


def load_model(checkpoint_path, result_path, hparams, progress, mod, prog=False):
    checkpoint_file = checkpoint_path

    if not prog:
        param_file = open(result_path, 'r')
        check_epoch = int(checkpoint_file.split("epoch=", 1)[1].split('-', 1)[0])
        resultjsons = param_file.read().split('\n')

    if mod == 'RC':
        if prog:
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
        else:
            results = json.loads(resultjsons[check_epoch + 1])
            params = results['config']
            lr = params['lr']
            dr = params['dr']
            batch_size = params['batch_size']
            datatype = params['datatype']
            loss = results['loss']
            acc = results['acc']

        con = {'lr': lr, 'dr': dr, 'batch_size': batch_size, 'datatype': datatype}
        model = RC(con, 2, 152, optimizer='adam')

        # checkpoint = torch.load(checkpoint_file)
        # model.prepare_data()
        # model.criterion.weight = torch.tensor([0., 0.])  # need to add as this is saved by the checkpoint file
        # model.load_state_dict(checkpoint['state_dict'])

        model.eval()
    elif mod == 'RCF':
        if prog:
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
        else:
            results = json.loads(resultjsons[check_epoch + 1])
            params = results['config']
            lr = params['lr']
            dr = params['dr']
            batch_size = params['batch_size']
            datatype = params['datatype']
            loss = results['loss']
            acc = results['acc']

        con = {'lr': lr, 'dr': dr, 'batch_size': batch_size, 'datatype': datatype}
        model = RCF(con, 2, 18, optimizer='adam')

        # checkpoint = torch.load(checkpoint_file)
        # model.prepare_data()
        # model.criterion.weight = torch.tensor([0., 0.])  # need to add as this is saved by the checkpoint file
        # model.load_state_dict(checkpoint['state_dict'])

        model.eval()
    elif mod == "VAES":
        results = json.loads(resultjsons[check_epoch + 1])
        params = results['config']
        lr = params['lr']
        dr = params['dr']
        batch_size = params['batch_size']
        datatype = params['datatype']
        z_dim = params['z_dim']
        loss = results['loss']
        acc = results['acc']

        con = {'lr': lr, 'dr': dr, 'batch_size': batch_size, 'z_dim': z_dim, 'datatype': datatype}
        model = VAES(con, True, 152, image_channels=1, hidden_dims=[128, 128, 128, 128, 128], out_image_channels=1, output_size=2, fcl_layers=[])

        # checkpoint = torch.load(checkpoint_file)
        # model.criterion.weight = torch.tensor([0., 0.])  # need to add as this is saved by the checkpoint file
        # model.load_state_dict(checkpoint['state_dict'])

        model.eval()
    elif mod == "VAEL":
        results = json.loads(resultjsons[check_epoch + 1])
        params = results['config']
        lr = params['lr']
        dr = params['dr']
        batch_size = params['batch_size']
        datatype = params['datatype']
        z_dim = params['z_dim']
        loss = results['loss']
        acc = results['acc']

        con = {'lr': lr, 'dr': dr, 'batch_size': batch_size, 'z_dim': z_dim, 'datatype': datatype}
        model = VAEL(con, True, 152, image_channels=1, hidden_dims=[128, 128, 128, 128, 128], out_image_channels=1, output_size=2, fcl_layers=[])

        # checkpoint = torch.load(checkpoint_file)
        # model.criterion.weight = torch.tensor([0., 0.])  # need to add as this is saved by the checkpoint file
        # model.load_state_dict(checkpoint['state_dict'])
        #
        # model.eval()
    elif mod == "SC":
        results = json.loads(resultjsons[check_epoch + 1])
        params = results['config']
        lr = params['lr']
        dr = params['dr']
        batch_size = params['batch_size']
        datatype = params['datatype']
        distance_cutoff = params['distance_cutoff']
        loss = results['loss']
        acc = results['acc']

        con = {'lr': lr, 'dr': dr, 'batch_size': batch_size, 'datatype': datatype, 'distance_cutoff': distance_cutoff}
        # model = ResNetClassifier(con, 2, 152, optimizer='adam')
        model = SC(con, outputs=2)

        # checkpoint = torch.load(checkpoint_file)
        # model.prepare_data()
        # # model.criterion.weight = torch.tensor([0., 0.]) # need to add as this is saved by the checkpoint file
        # model.load_state_dict(checkpoint['state_dict'])
        #
        # model.eval()

    checkpoint = torch.load(checkpoint_file)
    model.prepare_data()
    model.criterion.weight = torch.tensor([0., 0.])  # need to add as this is saved by the checkpoint file
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model, loss, acc

def predict_res(model, dataloader):
    vd = dataloader
    yt, yp = [], []
    for i, batch in enumerate(vd):
        seqs, ohe, labels = batch
        softmax = model(ohe)

        preds = torch.argmax(softmax, 1).clone().double()  # convert to torch float 64
        predcpu = list(preds.detach().cpu().numpy())
        ycpu = labels
        ver_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)

        # Make confusion Matrix
        y_true = ycpu.detach().cpu().numpy().astype('bool').tolist()
        y_pred = np.asarray(predcpu, dtype=np.bool).tolist()
        yt += y_true
        yp += y_pred
    return yt, yp

def predict_vae(model, dataloader):
    for i, batch in enumerate(dataloader):
        seqs, ohe, labels = batch
        seq_aves = []
        pred_aves = []
        replicas = 25
        for _ in range(replicas):
            predictions, xp, mu, logvar = model(ohe)
            seq_aves.append(xp)
            pred_aves.append(predictions)
        predictions = torch.mean(torch.stack(pred_aves, dim=0), dim=0)
        xp = torch.mean(torch.stack(seq_aves, dim=0), dim=0)
        xpp = torch.where(xp > 0.5, 1.0, 0.0)
        recon_acc = (ohe == xpp).float().mean()
        ver_seq_acc = recon_acc.item()

        preds = torch.argmax(predictions, 1).clone().double()  # convert to torch float 64
        predcpu = list(preds.detach().cpu().numpy())
        ycpu = labels
        # ver_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)
        # Make confusion Matrix
        y_true = ycpu.detach().cpu().numpy().astype('bool').tolist()
        y_pred = np.asarray(predcpu, dtype=np.bool).tolist()
    return y_true, y_pred

def sia_collate(batch):
    seqs1 = np.vstack([item[0] for item in batch])
    seqs2 = np.vstack([item[1] for item in batch])
    one_hots1 = torch.from_numpy(np.vstack([item[2] for item in batch]))
    one_hots2 = torch.from_numpy(np.vstack([item[3] for item in batch]))
    labels = torch.from_numpy(np.vstack([item[4] for item in batch])).squeeze(1)  # torch.LongTensor(target)
    return [seqs1, seqs2, one_hots1, one_hots2, labels]

def predict_sia(model, dataloader, dc):
    tmp_va = []
    preds = []
    truth = []
    for i, batch in enumerate(dataloader):
        seq1, seq2, x1, x2, y = batch
        out1, out2 = model(x1, x2)
        # val_loss = self.criterion(out1, out2, y)
        euclidean_distance = F.pairwise_distance(out1, out2)
        # print("distance", euclidean_distance)
        # self.val_dists.append(euclidean_distance)
        tmp = 1. if euclidean_distance.cpu().detach().numpy() < dc else 0
        # pred = torch.tensor([1.]) if euclidean_distance < torch.tensor([self.distance_cutoff]) else torch.tensor([0.])
        preds.append(tmp)
        va = 1. if tmp == y.cpu().detach().numpy() else 0.  # we'll try this
        tmp_va.append(va)
        truth.append(bool(y.cpu().detach()))

    # Make confusion Matrix
    y_true = truth
    y_pred = np.asarray(preds, dtype=np.bool).tolist()
    return y_true, y_pred

def prob_res(model, dataloader):
    vd = dataloader
    yt, yp = [], []
    probs = []
    for i, batch in enumerate(vd):
        seqs, ohe, labels = batch
        softmax = model(ohe)

        preds = torch.argmax(softmax, 1).clone().double()  # convert to torch float 64
        predcpu = list(preds.detach().cpu().numpy())
        ycpu = labels
        y_true = ycpu.detach().cpu().numpy().astype('bool').tolist()
        yt += y_true
        probs += list(softmax.detach().cpu().numpy().tolist())
    # print(probs)
    # print(yt)
    return yt, probs

def prob_vae(model, dataloader):
    yt, yp = [], []
    probs = []
    for i, batch in enumerate(dataloader):
        seqs, ohe, labels = batch
        seq_aves = []
        pred_aves = []
        replicas = 25
        for _ in range(replicas):
            predictions, xp, mu, logvar = model(ohe)
            seq_aves.append(xp)
            pred_aves.append(predictions)
        predictions = torch.mean(torch.stack(pred_aves, dim=0), dim=0)
        xp = torch.mean(torch.stack(seq_aves, dim=0), dim=0)
        xpp = torch.where(xp > 0.5, 1.0, 0.0)
        recon_acc = (ohe == xpp).float().mean()
        ver_seq_acc = recon_acc.item()

        preds = torch.argmax(predictions, 1).clone().double()  # convert to torch float 64
        predcpu = list(preds.detach().cpu().numpy())
        ycpu = labels
        # ver_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)
        # Make confusion Matrix
        y_true = ycpu.detach().cpu().numpy().astype('bool').tolist()
        y_pred = np.asarray(predcpu, dtype=np.bool).tolist()
        yt += y_true
        probs += list(predictions.detach().cpu().numpy().tolist())
    return yt, probs

def dist_sia(model, dataloader, dc):
    tmp_va = []
    preds = []
    truth = []
    yt, yp = [], []
    dists = []
    for i, batch in enumerate(dataloader):
        seq1, seq2, x1, x2, y = batch
        out1, out2 = model(x1, x2)
        # val_loss = self.criterion(out1, out2, y)
        euclidean_distance = F.pairwise_distance(out1, out2)
        # print("distance", euclidean_distance)
        # self.val_dists.append(euclidean_distance)
        tmp = 1. if euclidean_distance.cpu().detach().numpy() < dc else 0
        # pred = torch.tensor([1.]) if euclidean_distance < torch.tensor([self.distance_cutoff]) else torch.tensor([0.])
        preds.append(tmp)
        va = 1. if tmp == y.cpu().detach().numpy() else 0.  # we'll try this
        tmp_va.append(va)
        truth.append(bool(y.cpu().detach()))
        dists += euclidean_distance.detach().cpu().numpy().tolist()

    return truth, dists

def exp_results_check(checkpoint_path, hparams, progress, result_path, title, mod="none", prog=False):
    # example
    # checkpoint_file = './ray_results/tune_vae_asha/train_vae_a45d1_00000_0_batch_size=64,dr=0.029188,lr=0.0075796,z_dim=10_2021-07-13_12-50-57/checkpoints/epoch=28-step=15891.ckpt'

    # Load Model and Parameters
    model, loss, acc = load_model(checkpoint_path, result_path, hparams, progress, mod, prog=prog)

    # model loaded, now get data ready
    test_set = m.test_set_corr
    dca_set = m.dca_test_set_corr
    verdict = {'sequence':list(test_set.keys()), 'binary':list(test_set.values())}
    dcadict = {'sequence':list(dca_set.keys()), 'binary':list(dca_set.values())}

    _verification = pd.DataFrame(verdict)
    _dca = pd.DataFrame(dcadict)

    if mod == "SC":
        ver_reader = NAReader_Siamese_Verification(_verification, shuffle=False)
        dca_reader = NAReader_Siamese_Verification(_dca, shuffle=False)

        ver_loader = torch.utils.data.DataLoader(
            ver_reader,
            batch_size=1,
            collate_fn=sia_collate,
            # num_workers=4,
            # pin_memory=True,
            shuffle=False
        )

        dca_loader = torch.utils.data.DataLoader(
            dca_reader,
            batch_size=1,
            collate_fn=sia_collate,
            # num_workers=4,
            # pin_memory=True,
            shuffle=False
        )
    else:
        ver_reader = m.NAReader(_verification, shuffle=False)
        dca_reader = m.NAReader(_dca, shuffle=False)

        ver_loader = torch.utils.data.DataLoader(
            ver_reader,
            batch_size=len(ver_reader),
            collate_fn=m.my_collate,
            # num_workers=4,
            # pin_memory=True,
            shuffle=False
        )

        dca_loader = torch.utils.data.DataLoader(
            dca_reader,
            batch_size=len(dca_reader),
            collate_fn=m.my_collate,
            # num_workers=4,
            # pin_memory=True,
            shuffle=False
        )

    if mod == "RC" or mod == "RCF":
        yt_ver, yp_ver = predict_res(model, ver_loader)
        yt_dca, yp_dca = predict_res(model, dca_loader)
    elif mod == "VAES" or mod == "VAEL":
        yt_ver, yp_ver = predict_vae(model, ver_loader)
        yt_dca, yp_dca = predict_vae(model, dca_loader)
    elif mod == "SC":
        yt_ver, yp_ver = predict_sia(model, ver_loader, model.distance_cutoff)
        yt_dca, yp_dca = predict_sia(model, dca_loader, model.distance_cutoff)

    score_ver = np.asarray([1 if x == yp_ver[xid] else 0 for xid, x in enumerate(yt_ver)])
    score_dca = np.asarray([1 if x == yp_dca[xid] else 0 for xid, x in enumerate(yt_dca)])
    ver_acc = np.mean(score_ver)
    dca_acc = np.mean(score_dca)

    f1_ver = sklearn.metrics.f1_score(yt_ver, yp_ver)
    f1_dca = sklearn.metrics.f1_score(yt_dca, yp_dca)
    f1_mean = (f1_ver*23 + f1_dca*16)/(16+23)


    cm = sklearn.metrics.confusion_matrix(yt_ver, yp_ver, normalize='true')
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
    print("Validation Loss", loss, file=o)
    print("Validation Accuracy", acc, file=o)
    print("Verification Accuracy:", ver_acc, "of dataset size:", len(test_set.keys()), file=o)
    print("DCA Accuracy:", dca_acc, "of dataset size:", len(dca_set.keys()), file=o)
    print("F1-scores:", "ver:", f1_ver, "dca:", f1_dca, "MEAN F1:", f1_mean,  file=o)
    o.close()

def output_all_probabilities(checkpoint_path, hparams, progress, result_path, title, mod="none", prog=False):
    # example
    # checkpoint_file = './ray_results/tune_vae_asha/train_vae_a45d1_00000_0_batch_size=64,dr=0.029188,lr=0.0075796,z_dim=10_2021-07-13_12-50-57/checkpoints/epoch=28-step=15891.ckpt'
    # Load Model and Parameters
    model, loss, acc = load_model(checkpoint_path, result_path, hparams, progress, mod, prog=prog)

    model.prepare_data()
    traind = model.train_dataloader()
    vald = model.val_dataloader()

    # model loaded, now get data ready
    test_set = m.test_set_corr
    dca_set = m.dca_test_set_corr
    verdict = {'sequence':list(test_set.keys()), 'binary':list(test_set.values())}
    dcadict = {'sequence':list(dca_set.keys()), 'binary':list(dca_set.values())}

    _verification = pd.DataFrame(verdict)
    _dca = pd.DataFrame(dcadict)

    if mod == "SC":
        ver_reader = NAReader_Siamese_Verification(_verification, shuffle=False)
        dca_reader = NAReader_Siamese_Verification(_dca, shuffle=False)

        ver_loader = torch.utils.data.DataLoader(
            ver_reader,
            batch_size=1,
            collate_fn=sia_collate,
            # num_workers=4,
            # pin_memory=True,
            shuffle=False
        )

        dca_loader = torch.utils.data.DataLoader(
            dca_reader,
            batch_size=1,
            collate_fn=sia_collate,
            # num_workers=4,
            # pin_memory=True,
            shuffle=False
        )
    else:
        ver_reader = m.NAReader(_verification, shuffle=False)
        dca_reader = m.NAReader(_dca, shuffle=False)

        ver_loader = torch.utils.data.DataLoader(
            ver_reader,
            batch_size=len(ver_reader),
            collate_fn=m.my_collate,
            # num_workers=4,
            # pin_memory=True,
            shuffle=False
        )

        dca_loader = torch.utils.data.DataLoader(
            dca_reader,
            batch_size=len(dca_reader),
            collate_fn=m.my_collate,
            # num_workers=4,
            # pin_memory=True,
            shuffle=False
        )

    if mod == "RC" or mod == "RCF":
        yt_train, probs_train = prob_res(model, traind)
        yt_val, probs_val = prob_res(model, vald)
        yt_ver, probs_ver = prob_res(model, ver_loader)
        yt_dca, probs_dca = prob_res(model, dca_loader)
    elif mod == "VAES" or mod == "VAEL":
        yt_train, probs_train = prob_vae(model, traind)
        yt_val, probs_val = prob_vae(model, vald)
        yt_ver, probs_ver = prob_vae(model, ver_loader)
        yt_dca, probs_dca = prob_vae(model, dca_loader)
    elif mod == "SC":
        yt_train, probs_train = dist_sia(model, traind, model.distance_cutoff)
        yt_val, probs_val = dist_sia(model, vald, model.distance_cutoff)
        yt_ver, probs_ver = dist_sia(model, ver_loader, model.distance_cutoff)
        yt_dca, probs_dca = dist_sia(model, dca_loader, model.distance_cutoff)

    if mod != "SC":
        # data = {"dca_truth": yt_dca, "dca_prob": probs_dca}
        data = {"train_truth": yt_train, "train_prob": probs_train, "val_truth": yt_val, "val_prob": probs_val,
            "rbm_truth": yt_ver, "rbm_prob": probs_ver, "dca_truth": yt_dca, "dca_prob": probs_dca}
    else:
        # data = {"dca_truth": yt_dca, "dca_prob": probs_dca}
        data = {"train_truth": yt_train, "train_prob": probs_train, "val_truth": yt_val, "val_prob": probs_val,
                "rbm_truth": yt_ver, "rbm_prob": probs_ver, "dca_truth": yt_dca, "dca_prob": probs_dca, "distance_cutoff": model.distance_cutoff}

    with open('../../Dropbox (ASU)/Projects/Aptamer_ML/' + title + '.json', "w+") as f:
        json.dump(data, f)

    return
    # score_ver = np.asarray([1 if x == yp_ver[xid] else 0 for xid, x in enumerate(yt_ver)])
    # score_dca = np.asarray([1 if x == yp_dca[xid] else 0 for xid, x in enumerate(yt_dca)])
    # ver_acc = np.mean(score_ver)
    # dca_acc = np.mean(score_dca)
    #
    # f1_ver = sklearn.metrics.f1_score(yt_ver, yp_ver)
    # f1_dca = sklearn.metrics.f1_score(yt_dca, yp_dca)
    # f1_mean = (f1_ver*23 + f1_dca*16)/(16+23)
    #
    #
    # cm = sklearn.metrics.confusion_matrix(yt_ver, yp_ver, normalize='true')
    # df_cm = pd.DataFrame(cm, index=[0, 1], columns=[0, 1])
    # plt.figure(figsize=(10, 7))
    # ax = plt.subplot()
    # seaborn.set(font_scale=3.0)
    # seaborn.heatmap(df_cm, annot=True, ax=ax)
    # label_font = {'size': '26'}
    # ax.tick_params(axis='both', which='major', labelsize=40)
    # ax.xaxis.set_ticklabels(["0", "1"])
    # ax.yaxis.set_ticklabels(["0", "1"])
    # # plt.title(title)
    # plt.savefig('../../Dropbox (ASU)/Projects/Aptamer_ML/' + title)
    #
    # o = open('../../Dropbox (ASU)/Projects/Aptamer_ML/'+title+'results.txt', "w+")
    # print("Validation Loss", loss, file=o)
    # print("Validation Accuracy", acc, file=o)
    # print("Verification Accuracy:", ver_acc, "of dataset size:", len(test_set.keys()), file=o)
    # print("DCA Accuracy:", dca_acc, "of dataset size:", len(dca_set.keys()), file=o)
    # print("F1-scores:", "ver:", f1_ver, "dca:", f1_dca, "MEAN F1:", f1_mean,  file=o)
    # o.close()

#### Checkpoints and Titles
res_asha = ["/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_271b8_00000_0_batch_size=128,dr=0.023053,lr=0.0052977_2021-07-13_10-31-27/",
                "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_9b8d8_00004_4_batch_size=64,dr=0.059921,lr=0.00018311_2021-08-10_13-19-27/",
                "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_24aad_00001_1_batch_size=128,dr=0.13269,lr=0.00073351_2021-07-17_14-30-02/",
                "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_44bef_00005_5_batch_size=32,dr=0.013172,lr=0.00022575_2021-07-15_12-20-05/",
                "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_44c5e_00008_8_batch_size=128,dr=0.28844,lr=0.00047726_2021-07-15_11-57-44/",
                "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_04904_00004_4_batch_size=128,dr=0.099805,lr=0.00091394_2021-07-17_14-38-35/"]

res_asha_check = ["checkpoints/epoch=28-step=8235.ckpt",
               "checkpoints/epoch=28-step=15920.ckpt",
               "checkpoints/epoch=28-step=15340.ckpt",
               "checkpoints/epoch=28-step=37496.ckpt",
                "checkpoints/epoch=28-step=9279.ckpt",
               "checkpoints/epoch=28-step=18559.ckpt"]

res_asha_titles = ["Resnet ASHA Scheduler Right Arm",
    "Resnet ASHA Scheduler Left Arm",
    "Resnet ASHA Scheduler Both Arms",
    "Resnet ASHA Scheduler Generated Left Arm",
    "Resnet ASHA Scheduler Generated Right Arm",
    "Resnet ASHA Scheduler Generated Both Arms"]

res_asha_dict = ["res_asha_r",
    "res_asha_l",
    "res_asha_b",
    "res_asha_gl",
    "res_asha_gr",
    "res_asha_gb"]

res_bay = ["/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_d12af8ba_1_batch_size=32,datatype=HCRT,dr=0.1904,lr=0.095076_2021-07-26_15-28-35",
           "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_dd27d952_7_batch_size=32,datatype=HCLT,dr=0.41706,lr=0.021313_2021-07-26_22-31-19",
           "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_f5081252_9_batch_size=32,datatype=HCB20T,dr=0.1556,lr=0.052523_2021-08-03_21-36-14"]

res_bay_check = ["/checkpoints/epoch=48-step=55810.ckpt",
                   "/checkpoints/epoch=48-step=53801.ckpt",
                   "/checkpoints/epoch=8-step=19043.ckpt"]

res_bay_titles = ["Resnet Bayes Optimization Right Arm",
              "Resnet Bayes Optimization Left Arm",
              "Resnet Bayes Optimization Both Arms"]

res_bay_dict = ["res_bayes_r",
              "res_bayes_l",
              "res_bayes_b"]

res_fc_asha = ["/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_9424b_00007_7_batch_size=128,dr=0.026699,lr=0.0035935_2021-07-23_14-40-59",
               "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_9424c_00000_0_batch_size=32,dr=0.0059433,lr=0.006157_2021-07-23_14-11-42",
               "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_623cf_00004_4_batch_size=128,dr=0.0087122,lr=0.00017978_2021-08-03_10-55-18",
               "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_d0ca5_00003_3_batch_size=128,dr=0.29019,lr=0.00032726_2021-07-26_15-59-52",
               "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_d0ca5_00000_0_batch_size=32,dr=0.048774,lr=0.00011789_2021-07-26_15-28-35",
               "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_623ce_00001_1_batch_size=32,dr=0.17816,lr=0.00074892_2021-08-03_10-36-51"]

res_fc_check = ["/checkpoints/epoch=28-step=7945.ckpt",
               "/checkpoints/epoch=28-step=32972.ckpt",
               "/checkpoints/epoch=18-step=10050.ckpt",
               "/checkpoints/epoch=8-step=2879.ckpt",
               "/checkpoints/epoch=28-step=37525.ckpt",
               "/checkpoints/epoch=28-step=74210.ckpt"]

res_fc_titles = ["Resnet FC ASHA Scheduler Left Arm",
          "Resnet FC ASHA Scheduler Right Arm",
          "Resnet FC ASHA Scheduler Both Arms",
          "Resnet FC ASHA Scheduler Generated Left Arm",
          "Resnet FC ASHA Scheduler Generated Right Arm",
          "Resnet FC ASHA Scheduler Generated Both Arms"]

res_fc_dict = ["res_fc_asha_l",
          "res_fc_asha_r",
          "res_fc_asha_b",
          "res_fc_asha_gl",
          "res_fc_asha_gr",
          "res_fc_asha_gb"]

res_fc_bay = ["/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_66b50e64_3_batch_size=64,datatype=HCRT,dr=0.082229,lr=0.015684_2021-08-05_13-13-56",
                  "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_3111450e_9_batch_size=64,datatype=HCLT,dr=0.1556,lr=0.052523_2021-08-05_14-09-45",
                  "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_resnet_f1b92dc4_8_batch_size=64,datatype=HCB20T,dr=0.095003,lr=0.018422_2021-08-05_15-14-40"]

res_fc_bay_check = ["/checkpoints/epoch=48-step=27929.ckpt",
                       "/checkpoints/epoch=48-step=26851.ckpt",
                       "/checkpoints/epoch=48-step=51841.ckpt"]

res_fc_bay_titles = ["Resnet FC Bayes Optimization Right Arm",
                  "Resnet FC Bayes Optimization Left Arm",
                  "Resnet FC Bayes Optimization Both Arms"]

res_fc_bay_dict = ["res_fc_bayes_r",
                  "res_fc_bayes_l",
                  "res_fc_bayes_b"]

vae_short_asha = ["/mnt/D1/globus/new_vae_results/vae_short_hcrt/",
                      "/mnt/D1/globus/new_vae_results/vae_short_hclt/",
                      "/mnt/D1/globus/new_vae_results/vae_short_hcb20t/",
                      "/mnt/D1/globus/new_vae_results/vae_short_hcgb20t/",
                      "/mnt/D1/globus/new_vae_results/vae_short_hcgrt/",
                      "/mnt/D1/globus/new_vae_results/vae_short_hcglt/"]

vae_short_check = ["/checkpoints/epoch=28-step=8322.ckpt",
               "/checkpoints/epoch=28-step=15862.ckpt",
               "/checkpoints/epoch=28-step=30420.ckpt",
               "/checkpoints/epoch=28-step=18385.ckpt",
               "/checkpoints/epoch=28-step=9453.ckpt",
               "/checkpoints/epoch=28-step=9279.ckpt"]

vae_short_titles = ["VAE Short ASHA Scheduler Right Arm",
          "VAE Short ASHA Scheduler Left Arm",
          "VAE Short ASHA Scheduler Both Arms",
          "VAE Short ASHA Scheduler Generated Both Arms",
          "VAE Short ASHA Scheduler Generated Right Arm",
          "VAE Short ASHA Scheduler Generated Left Arm"]

vae_short_dict = ["vae_short_asha_r",
          "vae_short_asha_l",
          "vae_short_asha_b",
          "vae_short_asha_gb",
          "vae_short_asha_gr",
          "vae_short_asha_gl"]

vae_short_bay = ["/mnt/D1/globus/new_vae_results/vae_short_bayes_hcrt/",
                 "/mnt/D1/globus/new_vae_results/vae_short_bayes_hcb20t/",
                 "/mnt/D1/globus/new_vae_results/vae_short_bayes_hclt/"]

vae_short_bay_check= ["checkpoints/epoch=48-step=28076.ckpt",
                   "checkpoints/epoch=48-step=51400.ckpt",
                   "checkpoints/epoch=48-step=26802.ckpt"]

vae_short_bay_titles = ["VAE Short Bayes Optimization Right Arm",
              "VAE Short Bayes Optimization Both Arm",
              "VAE Short Bayes Optimization Left Arms"]

vae_short_bay_dict = ["vae_short_bayes_r",
              "vae_short_bayes_b",
              "vae_short_bayes_l"]

vae_long_asha = ["/mnt/D1/globus/new_vae_results/vae_long_hcrt/",
                      "/mnt/D1/globus/new_vae_results/vae_long_hclt/",
                      "/mnt/D1/globus/new_vae_results/vae_long_hcb20t/",
                      "/mnt/D1/globus/new_vae_results/vae_long_hcgb20t/",
                      "/mnt/D1/globus/new_vae_results/vae_long_hcgrt/",
                      "/mnt/D1/globus/new_vae_results/vae_long_hcglt/"]

vae_long_check = ["checkpoints/epoch=48-step=56153.ckpt",
               "checkpoints/epoch=48-step=53605.ckpt",
               "checkpoints/epoch=48-step=51400.ckpt",
               "checkpoints/epoch=48-step=31065.ckpt",
               "checkpoints/epoch=48-step=15973.ckpt",
               "checkpoints/epoch=48-step=31310.ckpt"]

vae_long_titles = ["VAE Long ASHA Scheduler Right Arm",
          "VAE Long ASHA Scheduler Left Arm",
          "VAE Long ASHA Scheduler Both Arms",
          "VAE Long ASHA Scheduler Generated Both Arms",
          "VAE Long ASHA Scheduler Generated Right Arm",
          "VAE Long ASHA Scheduler Generated Left Arm"]

vae_long_dict = ["vae_long_asha_r",
          "vae_long_asha_l",
          "vae_long_asha_b",
          "vae_long_asha_gb",
          "vae_long_asha_gr",
          "vae_long_asha_gl"]

vae_long_bay = ["/mnt/D1/globus/new_vae_results/vae_long_bayes_hcrt/",
                "/mnt/D1/globus/new_vae_results/vae_long_bayes_hclt/",
                "/mnt/D1/globus/new_vae_results/vae_long_bayes_hcb20t/"]

vae_long_bay_check = ["checkpoints/epoch=48-step=28076.ckpt",
                   "checkpoints/epoch=48-step=26802.ckpt",
                   "checkpoints/epoch=48-step=51400.ckpt"]

vae_long_bay_titles = ["VAE Long Bayes Optimization Right Arm",
              "VAE Long Bayes Optimization Left Arm",
              "VAE Long Bayes Optimization Both Arms"]

vae_long_bay_dict = ["vae_long_bayes_r",
              "vae_long_bayes_l",
              "vae_long_bayes_b"]

siamese_pbt = ["/mnt/D1/globus/agave_raw_results/best_checkpoints/train_siamese_954e2_00003_3_distance_cutoff=0.79521,dr=0.044483,lr=0.00012784_2021-07-17_14-33-11",
               "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_siamese_2511a_00007_7_distance_cutoff=1.7331,dr=0.011652,lr=0.00085261_2021-07-17_14-43-07",
               "/mnt/D1/globus/agave_raw_results/best_checkpoints/train_siamese_69ff7_00002_2_distance_cutoff=3.151,dr=0.015158,lr=0.09049_2021-08-09_13-20-34"]

siamese_pbt_check = ["/checkpoints/epoch=28-step=39151.ckpt",
               "/checkpoints/epoch=28-step=38807.ckpt",
               "/checkpoints/epoch=28-step=69423.ckpt"]

siamese_pbt_titles = ["Siamese PBT Scheduler Right Arm",
          "Siamese PBT Scheduler Left Arm",
          "Siamese PBT Scheduler Both Arms"]

siamese_pbt_dict = ["sia_pbt_r",
          "sia_pbt_l",
          "sia_pbt_b"]



#### Resnet
# for id, x in enumerate(res_bay):
#     # if id > 0:
#     #    break
#     check = x + res_bay_check[id]
#     res = x + "/result.json"
#     param = x + "params.json"
#     progress = x + "progress.csv"
#     exp_results_check(check, param, progress, res, res_bay_dict[id], mod="RC", prog=False)
#     # output_all_probabilities(check, param, progress, res, res_bay_dict[id], mod="RC", prog=False)
#
# for id, x in enumerate(res_asha):
#     # if id > 0:
#     #    break
#     check = x + res_asha_check[id]
#     res = x + "result.json"
#     param = x + "params.json"
#     progress = x + "progress.csv"
#
#     if id == 3 or id == 4:
#         pass
#         exp_results_check(check, param, progress, res, res_asha_dict[id], mod="RC", prog=True)
#         # val_results_check(check, param, progress, res, titles[id], r=False)
#         # output_all_probabilities(check, param, progress, res, res_asha_dict[id], mod="RC", prog=True)
#     else:
#         exp_results_check(check, param, progress, res, res_asha_dict[id], mod="RC", prog=False)
#         # output_all_probabilities(check, param, progress, res, res_asha_dict[id], mod="RC", prog=False)



#### Resnet FC
# for id, x in enumerate(res_fc_asha):
#     # if id > 0:
#     #    break
#     check = x + res_fc_check[id]
#     res = x + "/result.json"
#     params = x + "/params.json"
#     progress = x + "/progress.csv"
#
#     if id == 4:
#         exp_results_check(check, params, progress, res, res_fc_dict[id], mod="RCF", prog=True)
#         # output_all_probabilities(check, params, progress, res, res_fc_dict[id], mod="RCF", prog=True)
#     else:
#         exp_results_check(check, params, progress, res, res_fc_dict[id], mod="RCF", prog=False)
#         # output_all_probabilities(check, params, progress, res, res_fc_dict[id], mod="RCF", prog=False)
# #
# for id, x in enumerate(res_fc_bay):
#     # if id > 0:
#     #    break
#     check = x + res_fc_bay_check[id]
#     res = x + "/result.json"
#     params = x + "/params.json"
#     progress = x + "/progress.csv"
#     exp_results_check(check, params, progress, res, res_fc_bay_dict[id], mod="RCF", prog=False)
#     # output_all_probabilities(check, params, progress, res, res_fc_bay_dict[id], mod="RCF", prog=False)



#### VAE Short
# for id, x in enumerate(vae_short_asha):
#     # if id > 0:
#     #    break
#     check = x + vae_short_check[id]
#     res = x + "/result.json"
#     params = x + "/params.json"
#     progress = x + "/progress.csv"
#     exp_results_check(check, params, progress, res, vae_short_dict[id], mod="VAES", prog=False)
#     # output_all_probabilities(check, params, progress, res, vae_short_dict[id], mod="VAES", prog=False)
# #
# for id, x in enumerate(vae_short_bay):
#     # if id > 0:
#     #    break
#     check = x + vae_short_bay_check[id]
#     res = x + "/result.json"
#     params = x + "/params.json"
#     progress = x + "/progress.csv"
#     exp_results_check(check, params, progress, res, vae_short_bay_dict[id], mod="VAES", prog=False)
#     # output_all_probabilities(check, params, progress, res, vae_short_bay_dict[id], mod="VAES", prog=False)



#### VAE Long
# for id, x in enumerate(vae_long_asha):
#     # if id > 0:
#     #    break
#     check = x + vae_long_check[id]
#     res = x + "/result.json"
#     params = x + "/params.json"
#     progress = x + "/progress.csv"
#     exp_results_check(check, params, progress, res, vae_long_dict[id], mod="VAEL", prog=False)
#     # output_all_probabilities(check, params, progress, res, vae_long_dict[id], mod="VAEL", prog=False)
# #
# for id, x in enumerate(vae_long_bay):
#     # if id > 0:
#     #    break
#     check = x + vae_long_bay_check[id]
#     res = x + "/result.json"
#     params = x + "/params.json"
#     progress = x + "/progress.csv"
#     exp_results_check(check, params, progress, res, vae_long_bay_dict[id], mod="VAEL", prog=False)
#     # output_all_probabilities(check, params, progress, res, vae_long_bay_dict[id], mod="VAEL", prog=False)



#### Siamese
# for id, x in enumerate(siamese_pbt):
#     # if id > 0:
#     #    break
#     check = x + siamese_pbt_check[id]
#     res = x + "/result.json"
#     params = x + "/params.json"
#     progress = x + "/progress.csv"
#     exp_results_check(check, params, progress, res, siamese_pbt_dict[id], mod="SC", prog=False)
#     # output_all_probabilities(check, params, progress, res, siamese_pbt_dict[id], mod="SC", prog=False)




# Parse result files for formatting into overleaf
def get_info(files):
    titles = []
    val_accs = []
    rbm_accs = []
    dca_accs = []
    rbm_f1s = []
    dca_f1s = []
    mean_f1s = []

    for x in files:
        o = open("/home/jonah/Dropbox (ASU)/Projects/Aptamer_ML/"+x+"results.txt", 'r')
        lines = o.readlines()
        o.close()
        titles.append(x)
        val_loss = lines[0].split()[2]
        val_accs.append(str(round(float(lines[1].split()[2]), 3)))
        rbm_accs.append(str(round(float(lines[2].split()[2]), 3)))
        dca_accs.append(str(round(float(lines[3].split()[2]), 3)))
        f1s = lines[4].split()
        rbm_f1s.append(str(round(float(f1s[2]), 3)))
        dca_f1s.append(str(round(float(f1s[4]), 3)))
        mean_f1s.append(str(round(float(f1s[7]), 3)))

    table = ""
    for i in range(len(titles)):
        table += "%s & %s & %s & %s & %s\n" % (titles[i], val_accs[i], rbm_accs[i], dca_accs[i], mean_f1s[i])

    print(table)


all_files = vae_long_dict + vae_short_dict + res_fc_dict + res_fc_bay_dict + res_asha_dict + res_bay_dict + vae_short_bay_dict + vae_long_bay_dict + siamese_pbt_dict
get_info(all_files)
