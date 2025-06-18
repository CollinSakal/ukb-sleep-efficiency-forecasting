import PIL.Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy
from scipy import interpolate
from torch.optim.optimizer import Optimizer
import re

def duplicate_stdout_to_file(fname):
    import sys
    class Logger(object):
        def __init__(self,fname):
            self.terminal = sys.stdout
            self.log = open(fname, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.log.flush()

    sys.stdout = Logger(fname)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def real_model(model):
    if type(model)==torch.nn.DataParallel or type(model)==torch.nn.parallel.DistributedDataParallel:
        return model.module
    return model

def print_accuracy(pred,y,print_str=True):
    result = {}
    result_str = ''
    for i in np.unique(y):
        result[i]=(np.sum(pred[y==i]==i)/np.sum(y==i))
        result_str += f'{i}: {np.sum(pred[y==i]==i)/np.sum(y==i)} '
    if print_str:
        print('Eval Accuracy: '+result_str)
        print(f'Mean accuracy: {np.sum(pred==y)/len(y)}')
    result['mean'] = np.sum(pred==y)/len(y)
    return result,{0:np.where((pred==1)*(y==0))[0],1:np.where((pred==0)*(y==1))[0]}

def print_auroc(pred,y,print_str=True,n_bins=10):
    from sklearn.metrics import roc_auc_score,average_precision_score
    from sklearn.calibration import calibration_curve
    auroc = roc_auc_score(y, pred)
    auprc = average_precision_score(y,pred)
    cali = calibration_curve(y,pred,n_bins=n_bins)
    if print_str:
        print(f'Eval AUROC: {auroc}, AUPRC {auprc}')
        print(f'Cali {cali}')
    return (auroc,auprc)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

def generate_cam_gradcam(model,data,target_layer='layers.5'):
    import torchcam
    input_shape = data.shape[1:]
    cam_extractor = torchcam.methods.GradCAMpp(model, input_shape=(1,*input_shape), target_layer=target_layer)
    out = model(data)
    cam_orig = cam_extractor(0, out)[0].detach().cpu()
    cam_interp = interp_1d(cam_orig,input_shape[1])
    return (out.squeeze(dim=1).detach().cpu(), torch.Tensor(cam_interp).cpu())

def interp_1d(data,out_shape,method='linear'):
    inshape = data.shape[1]
    x = np.linspace(0,inshape-1,num=inshape)
    xnew = np.linspace(0,inshape-1,num=out_shape)
    f = interpolate.interp1d(x, data)
    return f(xnew)

def interp_by_socre(model,x,y):
    embeddings_list = model.layers.forward(x)
    orig_y_list = torch.sigmoid(model(x))
    score_list = []
    true_y = []
    for i in range(y.shape[0]):
        true_y.append(-1 if y[i] == 0 else 1)
    for x_index in range(x.shape[0]):
        cur_socre = []
        embeddings = embeddings_list[x_index].unsqueeze(0)
        orig_y = orig_y_list[x_index]
        for i in range(embeddings.shape[2]):
            if i == 0: cur_embeddings = embeddings[:,:,1:]
            elif i == embeddings.shape[2] - 1: cur_embeddings = embeddings[:,:,:-1]
            else: cur_embeddings = torch.concat((embeddings[:,:,:i],embeddings[:,:,i+1:]),dim=-1)
            cur_y = torch.sigmoid(model.lstm_forward(cur_embeddings))[0]
            cur_socre.extend(((orig_y-cur_y) * true_y[x_index]).detach().cpu().numpy().tolist())
        score_list.append(cur_socre)
    cam_interp = interp_1d(np.array(score_list), x.shape[2])
    return (orig_y_list.squeeze(dim=1).detach().cpu(), torch.Tensor(cam_interp).cpu())
'''
def interp_lime(model,train_x_orig,train_y,val_x_orig):
    dev = val_x_orig.device
    train_x = model.layers.forward(train_x_orig)
    val_x= model.layers.forward(val_x_orig.unsqueeze(0))
    train_x = train_x.detach().cpu().numpy()
    val_x= val_x.detach().cpu().numpy()
    train_y = train_y.detach().cpu().numpy()
    def forward(x):
        x = torch.Tensor(x).to(dev).chunk(50,0)
        ret = []
        for i in range(50):
            pred = torch.sigmoid(model.lstm_forward(x[i])).detach().cpu().numpy().flatten().tolist()
            ret.extend(pred)
        return np.array(ret)
    from lime.lime_tabular import RecurrentTabularExplainer
    explainer = RecurrentTabularExplainer(train_x,training_labels=train_y,feature_names=[f'timestep{i}' for i in range(val_x.shape[2])])
    explaination = explainer.explain_instance(val_x,forward,num_features=val_x.shape[2])
    explaination.as_pyplot_figure()
'''
