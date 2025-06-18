import datetime
import random
import sys
import os

import pandas
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from utils import *
from data import ACCDataset_DDP
from model import *
from train import train_ddp
def _bn_apply(x):
    if isinstance(x, nn.SyncBatchNorm):
        x.requires_grad_(False)
def _lstm_apply(x):
    if isinstance(x, nn.LSTM):
        x.requires_grad_(True)
        x.train()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simclr_mri0_adni: train')
    parser.add_argument('program')
    parser.add_argument('--load-model', dest='load_model', action='store', default=None, type=str)
    parser.add_argument('--data-path', dest='data_path', action='store', default='/root/workspace/PycharmProjects/AccelerometerSleep/data/acc/', type=str)
    parser.add_argument('--cv-train', dest='cv_train', action='store', type=str)
    parser.add_argument('--cv-val', dest='cv_val', action='store', type=str)
    parser.add_argument('--cv-train-save', dest='cv_train_save', action='store', type=str)
    parser.add_argument('--cv-val-save', dest='cv_val_save', action='store', type=str)
    parser.add_argument('--cv-gradcam-save', dest='cv_gradcam_save', action='store', type=str)
    parser.add_argument('--dataset', dest='dataset', action='store', type=str)
    parser.add_argument("--device", dest='device', action="store", type=int)
    parser.add_argument('--predict-type', dest='predict_type', action='store', default=None, type=str)
    parser.add_argument('--batch-size', dest='batch_size', action='store', default=100, type=int)

    args = parser.parse_args(sys.argv)
    torch.cuda.set_device(args.device)
    device = torch.device(args.device)
    print(args)
    dataset_path_list = {
        'acc-0hrs': (args.data_path + 'df-labels.csv',
                     args.data_path + 'acc-files-0hrs/', '-acc-0hrs.csv'),
        'acc-4hrs': (args.data_path + 'df-labels.csv',
                     args.data_path + 'acc-files-4hrs/', '-acc-4hrs.csv'),
        'acc-8hrs': (args.data_path + 'df-labels.csv',
                     args.data_path + 'acc-files-8hrs/', '-acc-8hrs.csv'),
    }
    dataset_path = dataset_path_list[args.dataset]
    label_name = args.predict_type

    cv_train_orig = pd.read_csv(args.cv_train)
    cv_val_orig = pd.read_csv(args.cv_val)

    cv_train_orig['PATH'] = dataset_path[1] + cv_train_orig.apply(lambda x: str(x['eid'].astype(int)), axis=1) + dataset_path[2]
    cv_train_orig['LABEL'] = cv_train_orig[label_name]

    cv_val_orig['PATH'] = dataset_path[1] + cv_val_orig.apply(lambda x: str(x['eid'].astype(int)), axis=1) + dataset_path[2]
    cv_val_orig['LABEL'] = cv_val_orig[label_name]

    #needed_eid = pd.DataFrame({'eid':[1000065,1000098,1000292,1000348,1000469,1000560,1000587,1000677,1000715,1000743]})
    #cv_train_orig = pd.merge(cv_train_orig,needed_eid,how='inner',on='eid')
    #cv_val_orig = pd.merge(cv_val_orig,needed_eid,how='inner',on='eid')

    cv_train = cv_train_orig.copy(True)
    cv_val = cv_val_orig.copy(True)

    trainset = ACCDataset_DDP(cv_train)
    valset = ACCDataset_DDP(cv_val)

    trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_size=args.batch_size, pin_memory=False, num_workers=16, drop_last=False)
    valloader = torch.utils.data.DataLoader(valset, shuffle=False, batch_size=args.batch_size, pin_memory=False, num_workers=16, drop_last=False)

    model = torch.load(args.load_model).to(device)
    model.eval()
    model.apply(_lstm_apply)
    flops = FlopCountAnalysis(model, torch.rand(1, 1, 83520).float().to(device))
    print(flop_count_table(flops))

    gradcam_list = []
    train_predict = []
    val_predict = []

    for step, data in enumerate(trainloader):
        y = data[0]
        x = data[1]
        x = x.to(device).float()
        y = y.to(device).float()
        pred, cam = generate_cam_gradcam(model, x, 'layers.10')
        #pred, cam = interp_by_socre(model,x,y)
        train_predict.append(pred)
        gradcam_list.append(cam)
    for step, data in enumerate(valloader):
        y = data[0]
        x = data[1]
        x = x.to(device).float()
        y = y.to(device).float()
        pred, cam = generate_cam_gradcam(model, x, 'layers.10')
        #pred, cam = interp_by_socre(model,x,y)
        val_predict.append(pred)
        gradcam_list.append(cam)

    train_predict = torch.sigmoid(torch.cat(train_predict,dim=0)).tolist()
    val_predict = torch.sigmoid(torch.cat(val_predict,dim=0)).tolist()
    gradcam_list = torch.cat(gradcam_list, dim=0).numpy()

    cv_train_orig['cnnlstm_pred'] = train_predict
    cv_val_orig['cnnlstm_pred'] = val_predict

    cv_train_orig.to_csv(args.cv_train_save,index=False)
    cv_val_orig.to_csv(args.cv_val_save,index=False)

    for i in range(len(cv_train_orig)):
        np.savetxt(args.cv_gradcam_save+str(int(cv_train_orig.iloc[i]['eid']))+'.csv',gradcam_list[i])
    for i in range(len(cv_val_orig)):
        np.savetxt(args.cv_gradcam_save+str(int(cv_val_orig.iloc[i]['eid']))+'.csv',gradcam_list[i + len(cv_train_orig)])
