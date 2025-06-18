import torch
from torch import nn
from torch import functional as F
class ACC_CNN_Model0(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(ACC_CNN_Model0, self).__init__(**kwargs)
        self.layers = nn.Sequential(
            nn.Conv1d(in_features, 64,kernel_size=10,stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128,kernel_size=10,stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=10, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 512, kernel_size=10, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(2),
        )
        self.rnn = nn.Sequential(
            nn.LSTM(input_size=512,hidden_size=512,num_layers=4,batch_first=True),
        )
        self.linear = nn.Sequential(
            nn.Linear(512,128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.transpose(x,1,2)
        x = self.rnn(x)[0][:,-1]
        x = self.linear(x)
        return x

class ACC_CNN_Model1(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(ACC_CNN_Model1, self).__init__(**kwargs)
        self.layers = nn.Sequential(
            nn.Conv1d(in_features, 64,kernel_size=10,stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(5),

            nn.Conv1d(64, 128,kernel_size=10,stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(5),

            nn.Conv1d(128, 256, kernel_size=10, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(5),

        )
        self.rnn = nn.Sequential(
            nn.LSTM(input_size=256,hidden_size=256,num_layers=1,batch_first=True),
        )
        self.linear = nn.Sequential(
            nn.Linear(256,64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.transpose(x,1,2)
        x = self.rnn(x)[0][:,-1]
        x = self.linear(x)
        return x

class ACC_CNN_Model2(nn.Module):
    def __init__(self, in_features, out_features,dropout,sigmoid,**kwargs):
        super(ACC_CNN_Model2, self).__init__(**kwargs)
        self.layers = nn.Sequential(
            nn.Conv1d(in_features, 64,kernel_size=10,stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(dropout),
            nn.MaxPool1d(3),

            nn.Conv1d(64, 128,kernel_size=10,stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout1d(dropout),
            nn.MaxPool1d(3),

            nn.Conv1d(128, 256, kernel_size=10, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout1d(dropout),

            nn.Conv1d(256, 512, kernel_size=10, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout1d(dropout),
        )
        self.rnn = nn.Sequential(
            nn.LSTM(input_size=512,hidden_size=512,num_layers=1,batch_first=True,dropout=dropout),
        )
        self.linear = nn.Sequential(
            nn.Linear(512,64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 1),
            nn.Sigmoid() if sigmoid else nn.Identity(),
        )
    def lstm_forward(self,x):
        x = torch.transpose(x,1,2)
        x = self.rnn(x)[0][:,-1]
        x = self.linear(x)
        return x
    def forward(self, x):
        x = self.layers(x)
        x = torch.transpose(x,1,2)
        x = self.rnn(x)[0][:,-1]
        x = self.linear(x)
        return x

class ACC_CNN_Model_ResNet(nn.Module):
    def __init__(self, in_features, out_features,dropout,sigmoid,**kwargs):
        super(ACC_CNN_Model_ResNet, self).__init__(**kwargs)
        from resnet import BasicBlock,conv1x1x1
        self.layers = nn.Sequential(
            nn.Conv1d(in_features, 64,kernel_size=10,stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(dropout),
            nn.MaxPool1d(3),
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),

            BasicBlock(64,128,stride=3,downsample=nn.Sequential(
                    conv1x1x1(64, 128, 3),
                    nn.BatchNorm1d(128))),
            BasicBlock(128, 128, stride=1),
            BasicBlock(128, 128, stride=1),
            nn.Dropout1d(dropout),
            nn.MaxPool1d(3),

            BasicBlock(128,256,stride=3,downsample=nn.Sequential(
                    conv1x1x1(128, 256, 3),
                    nn.BatchNorm1d(256))),
            BasicBlock(256, 256, stride=1),
            BasicBlock(256, 256, stride=1),
            nn.Dropout1d(dropout),
            nn.MaxPool1d(3),

            BasicBlock(256,512,stride=3,downsample=nn.Sequential(
                    conv1x1x1(256, 512, 3),
                    nn.BatchNorm1d(512))),
            BasicBlock(512, 512, stride=1),
            BasicBlock(512, 512, stride=1),
            nn.Dropout1d(dropout),
        )
        self.rnn = nn.Sequential(
            nn.LSTM(input_size=512,hidden_size=512,num_layers=1,batch_first=True,dropout=dropout),
        )
        self.linear = nn.Sequential(
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 1),
            nn.Sigmoid() if sigmoid else nn.Identity(),
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.transpose(x,1,2)
        x = self.rnn(x)[0][:,-1]
        x = self.linear(x)
        return x

class ACC_CNN_Model3(nn.Module):
    def __init__(self, in_features, out_features,dropout,sigmoid,**kwargs):
        super(ACC_CNN_Model3, self).__init__(**kwargs)
        self.layers = nn.Sequential(
            nn.Conv1d(in_features, 64,kernel_size=10,stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(dropout),
            nn.Conv1d(64, 64, kernel_size=10, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout1d(dropout),
            nn.MaxPool1d(3),

            nn.Conv1d(64, 128,kernel_size=10,stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout1d(dropout),
            nn.Conv1d(128, 128, kernel_size=10, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout1d(dropout),
            nn.MaxPool1d(3),

            nn.Conv1d(128, 256, kernel_size=10, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=10, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout1d(dropout),

            nn.Conv1d(256, 512, kernel_size=10, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=10, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout1d(dropout),
        )
        self.rnn = nn.Sequential(
            nn.LSTM(input_size=512,hidden_size=512,num_layers=1,batch_first=True,dropout=dropout),
        )
        self.linear = nn.Sequential(
            nn.Linear(512,64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 1),
            nn.Sigmoid() if sigmoid else nn.Identity(),
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.transpose(x,1,2)
        x = self.rnn(x)[0][:,-1]
        x = self.linear(x)
        return x