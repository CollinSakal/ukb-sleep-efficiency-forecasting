import datetime
import random
import sys
import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from utils import *
from data import ACCDataset_DDP
from model import *
from train import train_ddp
def setup(rank, world_size,port,seed):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    print(f"Bind to port: {os.environ['MASTER_PORT']}")
    # initialize the process group

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dist.init_process_group("nccl", rank=rank, world_size=world_size,timeout=datetime.timedelta(minutes=30))
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def partition_dataset(world_size,dataset_path,label_name,trainsize,valsize,testsize,cv_meta):

    if cv_meta[0] is None:
        meta = pd.read_csv(dataset_path[0]).sample(frac=1).reset_index(drop=True)
        meta['PATH'] = dataset_path[1] + meta.apply(lambda x: str(x['eid']), axis=1) + dataset_path[2]
        meta['LABEL'] = meta[label_name]

        left_size = len(meta) - valsize - testsize
        trainsize = trainsize if trainsize < left_size else left_size

        val = meta[0 : valsize]
        test = meta[valsize : valsize + testsize]
        train = meta[valsize + testsize : valsize + testsize + trainsize]
    else:
        cv_train = pd.read_csv(cv_meta[0])
        cv_val = pd.read_csv(cv_meta[1])

        cv_train['PATH'] = dataset_path[1] + cv_train.apply(lambda x: str(x['eid'].astype(int)), axis=1) + dataset_path[2]
        cv_train['LABEL'] = cv_train[label_name]

        cv_val['PATH'] = dataset_path[1] + cv_val.apply(lambda x: str(x['eid'].astype(int)), axis=1) + dataset_path[2]
        cv_val['LABEL'] = cv_val[label_name]

        train = cv_train
        val = cv_val
        test = None

    split = {
        'train': train,
        'val': val,
        'test': test,
    }

    return split
def setup_dataloader(rank,dataset,batch_size,world_size):
    trainset = ACCDataset_DDP(dataset['train'])
    sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=False, sampler=sampler)

    valloader = testloader = None
    if rank == 0:
        valset = ACCDataset_DDP(dataset['val'])
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                 shuffle=True, num_workers=0,pin_memory=True)

        testset = ACCDataset_DDP(dataset['test']) if dataset['test'] is not None else None
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=True, num_workers=0,pin_memory=True) if dataset['test'] is not None else None
    return (trainloader,valloader,testloader)

def main(rank,world_size,config):
    setup(rank,world_size,config['port'],config['seed'])
    if rank == 0:
        duplicate_stdout_to_file(config['save_path'] + 'log.txt')

    trainloader, valloader, testloader = setup_dataloader(rank,config['dataset_split'],config['batch_size'],world_size)

    if config['model'] == 'cnn':
        model = ACC_CNN_Model2(1,1,dropout=config['dropout'],sigmoid=False)
    elif config['model'] == 'cnns':
        model = ACC_CNN_Model2(1,1,dropout=config['dropout'],sigmoid=True)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    dist.barrier()
    print(count_parameters(model))

    optimizer = config['optimizer']([param for param in model.parameters() if param.requires_grad], lr=config['lr'])

    if config['loss_func'] == nn.BCEWithLogitsLoss:
        loss_func = config['loss_func'](pos_weight=torch.Tensor([4]).to(rank))
        type = 'classification'
    else:
        loss_func = config['loss_func']()
        type = 'regression'

    warmupscheduler = None
    mainscheduler = None
    if config['use_warmup_scheduler']:
        warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1, verbose=True)
    if config['use_main_scheduler']:
        mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config['t_0'], T_mult=config['t_mult'],eta_min=config['lr_min'],verbose=True)

    train_ddp(rank,config['epoch'],model,trainloader,valloader,testloader,optimizer,loss_func,config['save_path'],
                 warmupscheduler,mainscheduler,1,config['threshold'],config['warmup_steps'],type)
    cleanup()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='simclr_mri0_adni: train')
    parser.add_argument('program')
    parser.add_argument('--lr', dest='lr', action='store', default=1e-1, type=float)
    parser.add_argument('--lr-min', dest='lr_min', action='store', default=1e-2, type=float)
    parser.add_argument('--epoch', dest='epoch', action='store',default=500, type=int)
    parser.add_argument('--batch-size', dest='batch_size', action='store', default=10, type=int)
    parser.add_argument('--cosine-mult', dest='t_mult', action='store', default=1, type=int)
    parser.add_argument('--cosine-T', dest='t_0', action='store', default=20, type=int)
    parser.add_argument('--warmup-steps', dest='warmup_steps', action='store', default=10, type=int)
    parser.add_argument('--train-size', dest='train_size', action='store', default=-1, type=int)
    parser.add_argument('--val-size', dest='val_size', action='store',default=50, type=int)
    parser.add_argument('--test-size', dest='test_size', action='store',default=50, type=int)
    parser.add_argument('--group-ratio', dest='group_ratio', action='store', default=4.0, type=float)
    parser.add_argument('--threshold', dest='threshold', action='store',default=0.80, type=float)
    parser.add_argument('--predict-type', dest='predict_type', action='store', default=None, type=str)
    parser.add_argument('--load-model', dest='load_model', action='store', default=None, type=str)
    parser.add_argument('--model-type', dest='model_type', action='store', default='cnn', type=str)
    parser.add_argument('--dropout-rate', dest='dropout', action='store', default=0.2, type=float)
    parser.add_argument('--save-path', dest='save_path', action='store', default='/tmp/tmp_model', type=str)
    parser.add_argument('--data-path', dest='data_path', action='store', default='/root/workspace/PycharmProjects/AccelerometerSleep/data/acc/', type=str)
    parser.add_argument('--cv-train', dest='cv_train', action='store', type=str)
    parser.add_argument('--cv-val', dest='cv_val', action='store', type=str)
    parser.add_argument('--dataset', dest='dataset', action='store', type=str)
    parser.add_argument("--device", dest='device', action="extend", nargs="+", type=int)
    parser.add_argument('--comment', dest='comment', action='store', type=str)

    args = parser.parse_args(sys.argv)
    if not os.path.exists(args.save_path):
        os.system(f"mkdir -p {args.save_path}")
    duplicate_stdout_to_file(args.save_path +'/'+ 'init.txt')
    print(args)

    world_size = len(args.device)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(dev) for dev in args.device])

    dataset_path = {
        'acc-0hrs': (args.data_path + 'df-labels.csv',
                     args.data_path + 'acc-files-0hrs/', '-acc-0hrs.csv'),
        'acc-4hrs': (args.data_path + 'df-labels.csv',
                     args.data_path + 'acc-files-4hrs/', '-acc-4hrs.csv'),
        'acc-8hrs': (args.data_path + 'df-labels.csv',
                     args.data_path + 'acc-files-8hrs/', '-acc-8hrs.csv'),
    }
    predict_type_loss = {
        'sleep_efficiency': nn.MSELoss,
        'sleep_efficiency85': nn.BCEWithLogitsLoss,
        'sleep_efficiency90': nn.BCEWithLogitsLoss,
        'label': nn.BCEWithLogitsLoss,
    }
    dataset_split = partition_dataset(world_size,dataset_path[args.dataset],args.predict_type,args.train_size,args.val_size,args.test_size,(args.cv_train,args.cv_val))

    train_config = {
        'port': str(random.randint(50000,60000)),
        'dataset_split': dataset_split,
        'lr': args.lr,
        'lr_min': args.lr_min,
        'epoch': args.epoch,
        'threshold': args.threshold,
        'warmup_steps': args.warmup_steps,
        't_mult': args.t_mult,
        't_0': args.t_0,
        'dropout': args.dropout,
        'batch_size': args.batch_size // len(args.device),
        'save_path': args.save_path+'/',
        'eval_interval':1,
        'use_warmup_scheduler':True if args.load_model is None else False,
        'load_model': args.load_model,
        'use_main_scheduler':True,
        'optimizer':torch.optim.Adam,
        'model': args.model_type,
        'loss_func':predict_type_loss[args.predict_type],
        'seed': random.randint(1,10000)
    }

    mp.spawn(
        main,
        args=(world_size,train_config),
        nprocs=world_size,
    )
