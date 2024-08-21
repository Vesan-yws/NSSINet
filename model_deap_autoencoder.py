# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:03:57 2020

@author: user
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import adv_layer

class Autoencoder_imp_nssi(nn.Module):
    def __init__(self, hidden_dim, n_layer, n_filters, input_size,tripledomain):
        super(Autoencoder_imp_nssi, self).__init__()

        # Layer 1：conventional convolution for time dim
        self.conv1 = nn.Conv2d(1, int(16 * n_filters), (1, int(input_size / 2 + 1)), stride=1,
                               padding=(0, int(input_size / 4)))  # (16, C, T)
        self.batchNorm1 = nn.BatchNorm2d(int(16 * n_filters), False)
        self.n_filters = n_filters
        self.hidden_dim = hidden_dim
        self.length = input_size / 32
        # Layer 2：spatial convolution for channel dim
        self.depthwiseconv2 = nn.Conv2d(int(16 * n_filters), int(32 * n_filters), (63, 1), padding=0)  # (32, 1, T)
        self.batchNorm2 = nn.BatchNorm2d(int(32 * n_filters), False)  # 1*5000
        self.pooling1 = nn.MaxPool2d((1, 4), return_indices=True)

        # Layer 3： depthwise separable  convolutions
        self.separa1conv3 = nn.Conv2d(int(32 * n_filters), int(32 * n_filters), (1, int(input_size / 8 + 1)), stride=1,
                                      padding=(0, int(input_size / 16)), groups=int(32 * n_filters))  # (32, 1, T/5)
        self.separa2conv4 = nn.Conv2d(int(32 * n_filters), int(16 * n_filters), 1)  # (16, 1, T/4)
        self.batchNorm3 = nn.BatchNorm2d(int(16 * n_filters), False)
        self.pooling2 = nn.MaxPool2d((1, 8), return_indices=True)  # (16, 1, T/75)
        self.lstm = nn.LSTM(int(16 * n_filters), int(hidden_dim * n_filters), n_layer, batch_first=True,
                            bidirectional=True)
        # Layer 4：FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)
        self.dropout3 = nn.Dropout(p=0.25)
        self.dropout4 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(int(16 * n_filters), int(16 * n_filters))
        self.fc2 = nn.Linear(int(hidden_dim * 2 * n_filters), int(hidden_dim * n_filters))
        self.fc3 = nn.Linear(int(hidden_dim * n_filters), int(hidden_dim * 2 * n_filters))
        # Layer 1: FC Layer  reshape data which are from encoder
        self.fc4 = nn.Linear(int(2 * 16 * n_filters), int(16 * n_filters))
        # x = x.view((16,1,5))+
        # Layer 2:deconventional
        self.gru_en = nn.GRU(int(16 * n_filters), int(hidden_dim * n_filters), n_layer, batch_first=True,
                             bidirectional=True)
        self.gru_de = nn.GRU(int(2 * hidden_dim * n_filters), int(16 * n_filters), n_layer, batch_first=True,
                             bidirectional=True)
        self.unpooling2 = nn.MaxUnpool2d((1, 8))
        self.batchnorm4 = nn.BatchNorm2d(int(32 * n_filters), False)
        self.desepara2conv4 = nn.ConvTranspose2d(int(16 * n_filters), int(32 * n_filters), 1)
        self.desepara1conv3 = nn.ConvTranspose2d(int(32 * n_filters), int(32 * n_filters), (1, int(input_size / 8 + 1)),
                                                 stride=1, padding=(0, int(input_size / 16)),
                                                 groups=int(32 * n_filters))

        # Layer 3: de spatial convolution for channel dim
        self.unpooling1 = nn.MaxUnpool2d((1, 4))
        self.batchnorm5 = nn.BatchNorm2d(int(16 * n_filters), False)  #
        self.dedepthsepara1conv3 = nn.ConvTranspose2d(int(32 * n_filters), int(16 * n_filters), (63, 1), stride=1,
                                                      padding=0)

        # Layer 4: de spatial convolution for channel dim
        self.deconv1 = nn.ConvTranspose2d(int(16 * n_filters), 1, (1, int(input_size / 2 + 1)), stride=1,
                                          padding=(0, int(input_size / 4)))

        # self.fc = nn.Linear(int(16 * self.n_filters) * int(self.length), 64)
        self.classifier = nn.Linear(int(16 * self.n_filters) * int(self.length), 2)
        self.gender_classifier = adv_layer.GenderAdversarialLoss(hidden_1=int(16 * self.n_filters) * int(self.length))
        if tripledomain:
            self.domain_classifier = adv_layer.TripleDomainAdversarialLoss(hidden_1=int(16 * self.n_filters) * int(self.length))
        else:
            self.domain_classifier = adv_layer.DomainAdversarialLoss(hidden_1=int(16 * self.n_filters) * int(self.length))


    def forward(self, x, label_sex=None):
        # encoder
        x = self.conv1(x)
        x = self.batchNorm1(x)
        #            x = self.dropout4(x)
        # Layer 2
        x = self.depthwiseconv2(x)
        x = self.batchNorm2(x)
        x = F.elu(x)  ##激活函数
        x, idx2 = self.pooling1(x)  # get data and their index after pooling
        x = self.dropout1(x)
        # Layer 3
        x = self.separa1conv3(x)
        x = self.separa2conv4(x)
        x = self.batchNorm3(x)
        x = F.elu(x)
        x, idx3 = self.pooling2(x)

        # Layer 4：FC Layer
        x = x.permute(0, 3, 2, 1)
        x = x[:, :, -1, :, ]
        x = self.fc1(x)
        x = F.elu(x)
        #            x =self.dropout2(x)##有百分之多少的神经元可能会失活，用于防止过拟合
        out, _ = self.gru_en(x)
        x = out
        x = self.fc2(x)

        code = x.reshape((x.shape[0], int(16 * self.n_filters) * int(self.length)))

        # decoder
        x = self.fc3(x)
        out, _ = self.gru_de(x)
        x = out
        x = self.fc4(x)
        x = F.elu(x)
        #            x = self.dropout3(x)
        x = x.reshape((x.shape[0], x.shape[1], 1, x.shape[2]))
        x = x.permute(0, 3, 2, 1)
        x = self.unpooling2(x, idx3)

        x = self.desepara2conv4(x)
        x = self.desepara1conv3(x)
        x = self.batchnorm4(x)
        x = self.dropout4(x)

        # Layer 3
        x = F.elu(x)
        x = self.unpooling1(x, idx2)
        x = self.dedepthsepara1conv3(x)
        x = self.batchnorm5(x)
        # Layer 4
        x = self.deconv1(x)

        domain_output = None
        gender_output = None

        if self.training:
            domain_output = self.domain_classifier(code)
            gender_output = self.gender_classifier(code, label_sex)
        # pred1 = self.fc(code)
        # pred1 = F.elu(pred1)

        pred = self.classifier(code)

        return x, pred, domain_output, gender_output, code

class Discriminator_nssi(nn.Module):
    def __init__(self, n_layer, n_filters, input_size):
        super(Discriminator_nssi, self).__init__()
        self.conv1 = nn.Conv2d(1, int(8 * n_filters), (1, int(input_size / 2 + 1)), stride=1,
                               padding=(0, int(input_size / 4)))  # (16, C, T)
        self.batchNorm1 = nn.BatchNorm2d(8 * n_filters, False)
        self.length = input_size / 32
        # Layer 2：spatial convolution for channel dim
        self.depthwiseconv2 = nn.Conv2d(int(8 * n_filters), int(16 * n_filters), (63, 1), padding=0)  # (32, 1, T)
        self.batchNorm2 = nn.BatchNorm2d(int(16 * n_filters), False)  # 1*5000
        self.pooling1 = nn.MaxPool2d((1, 4), return_indices=False)
        self.separa1conv3 = nn.Conv2d(int(16 * n_filters), int(16 * n_filters), (1, int(input_size / 8 + 1)), stride=1,
                                      padding=(0, int(input_size / 16)), groups=int(16 * n_filters))
        self.separa2conv4 = nn.Conv2d(int(16 * n_filters), int(8 * n_filters), 1)  # (16, 1, T/4)
        self.batchNorm3 = nn.BatchNorm2d(int(8 * n_filters), False)
        self.pooling2 = nn.MaxPool2d((1, 8), return_indices=False)  # (16, 1, T/75)
        self.fc1 = nn.Linear(int(self.length * 8), 1)

    def forward(self, x):
        # encoder
        x = self.conv1(x)
        x = self.batchNorm1(x)
        #            x = self.dropout4(x)
        # Layer 2
        x = self.depthwiseconv2(x)
        x = self.batchNorm2(x)
        x = F.elu(x)  ##激活函数
        x = self.pooling1(x)  # get data and their index after pooling
        # Layer 3
        x = self.separa1conv3(x)
        x = self.separa2conv4(x)
        x = self.batchNorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)

        # Layer 4：FC Layer
        x = x.reshape((x.shape[0], int(self.length * 8)))
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

class Adversial_Loss_nssi(nn.Module):
    def __init__(self,lamda=1,input_size=512):
        super().__init__()
        self.Dis = Discriminator_nssi(1,1,input_size)
        self.lamda= lamda
        self.crition=nn.BCELoss()
    def forward(self,inputs,outputs):
        labelt_inp=torch.ones((inputs.shape[0],1)).cuda(0)
        labelt_out=torch.zeros((inputs.shape[0],1)).cuda(0)
        label_inp=self.Dis(inputs)
        label_out=self.Dis(outputs)
        loss_Dis=self.lamda*self.crition(torch.cat((label_inp,label_out),0),torch.cat((labelt_inp,labelt_out),0))
        return loss_Dis
