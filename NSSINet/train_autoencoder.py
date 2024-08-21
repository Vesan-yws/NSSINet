# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 14:18:59 2020

@author: user
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn import init
import os
import re
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from model_deap_autoencoder import Autoencoder_imp_nssi
from model_deap_autoencoder import Adversial_Loss_nssi
from sklearn import preprocessing
from scipy import signal
import random
from sklearn.metrics import confusion_matrix
sys_path = os.path.abspath("..")
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def get_weight(model):
    weight_list = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight = (name, param)
            weight_list.append(weight)
    return weight_list 

def regularization_loss(weight_list, weight_decay, p=2):

    L2_list=['conv1.weight',
             'depthwiseconv2.weight',
             'separa1conv3.weight',
             'separa2conv4.weight','fc_vector.weight',
             'vector_s']
    reg_loss=0
    for name, w in weight_list:
        if name in L2_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
    reg_loss=weight_decay*reg_loss
    return reg_loss

def train_discrinator(inputs,loss_fn,model,opti_loss):
    model.eval()
    opti_loss.zero_grad()
    with torch.no_grad():
        outputs,_,_,_,_=model(inputs)
    loss=loss_fn(inputs,outputs)
    loss.backward()
    opti_loss.step()
    return loss_fn,opti_loss

def weigth_init(m):
    setup_seed(20)
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)##对参数进行xavier初始化，为了使得网络中信息更好的流动，每一层输出的方差应该尽量相等
        init.constant_(m.bias.data,0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()    
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.03)
        m.bias.data.zero_()

def data_norm(X_data,resolution):
    X_data_sque=np.reshape(X_data,(np.shape(X_data)[0],63*resolution))
    X_data_sque_scaled=max_min_scale(X_data_sque)
    X_data=np.reshape(X_data_sque_scaled,(np.shape(X_data)[0],63,resolution))
    return X_data
def max_min_scale(data_sque):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    data_sque=min_max_scaler.fit_transform(data_sque)
    return data_sque

def get_dataset_nssi(norm_type,resolution,Session, s_index, u_index, test_idx, sexual_index):
    s_data_list=[]
    s_label_list=[]
    s_sex_label_list = []
    u_data_list = []
    u_label_list = []
    u_sex_label_list = []
    test_data_list = []
    test_label_list = []
    t_sex_label_list = []
    os.chdir(sys_path + '\\data_combine')
    path = 'Block' + str(Session)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    for info in os.listdir(path):
        domain = os.path.abspath(path)
        info_ = os.path.join(domain, info)

        sub_index_str_, _ = os.path.splitext(info)
        sub_index_str = re.findall(r'\d+', sub_index_str_)
        feature = scio.loadmat(info_)['Struct']['feature'][0, 0]
        label = scio.loadmat(info_)['Struct']['label'][0, 0]
        feature = signal.resample_poly(feature.T, 384, 500).T
        ######数据处理
        feature_two = feature.reshape(feature.shape[0], -1)
        feature_two = min_max_scaler.fit_transform(feature_two).astype('float32')
        feature = feature_two.reshape(feature.shape)[:, np.newaxis, :, :]

        one_hot_label_mat = np.zeros((len(label), 2))
        for i in range(len(label)):
            if label[i] == 0:
                one_hot_label = [1, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 2)
                one_hot_label_mat[i, :] = one_hot_label
            if label[i] == 1:
                one_hot_label = [0, 1]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 2)
                one_hot_label_mat[i, :] = one_hot_label

        sub_index = int(sub_index_str[0])-1
        sexual_label = sexual_index[sub_index]
        if sub_index in s_index:
            ## source labeled data
            feature_labeled = feature
            label_labeled = one_hot_label_mat
            s_data_list.append(feature_labeled)
            s_label_list.append(label_labeled)
            s_sex_label_list.append((np.tile(sexual_label, len(label_labeled)))[:, np.newaxis])


        if sub_index in u_index:
            ## source unlabeled data
            feature_unlabeled = feature
            label_unlabeled = one_hot_label_mat
            u_data_list.append(feature_unlabeled)
            u_label_list.append(label_unlabeled)
            u_sex_label_list.append((np.tile(sexual_label, len(label_unlabeled)))[:, np.newaxis])


        if sub_index in test_idx:
            ## target labeled data
            test_data_list.append(feature)
            test_label_list.append(one_hot_label_mat)
            t_sex_label_list.append((np.tile(sexual_label, len(one_hot_label_mat)))[:, np.newaxis])

    s_data, s_label, s_label_sex = np.vstack(s_data_list), np.vstack(s_label_list), np.vstack(s_sex_label_list)
    u_data, u_label, u_label_sex = np.vstack(u_data_list), np.vstack(u_label_list), np.vstack(u_sex_label_list)
    test_data, test_label, t_label_sex = np.vstack(test_data_list), np.vstack(test_label_list), np.vstack(t_sex_label_list)


    s_data_set = {'feature': s_data, 'label': s_label, 'label_sex': s_label_sex}
    u_data_set = {'feature': u_data, 'label': u_label, 'label_sex': u_label_sex}
    test_data_set = {'feature': test_data, 'label': test_label, 'label_sex': t_label_sex}
    return s_data_set, u_data_set, test_data_set

def model_validation_autoencoder(model,loader_valid):
    model.eval()
    loss=0
    test_acc_total, total_num = 0, 0
    loss_function=torch.nn.MSELoss(reduction='mean')
    label_list = []
    pred_list = []
    sex_list = []
    with torch.no_grad():
        setup_seed(20)
        for step, (data) in enumerate(loader_valid):
            inputs = Variable(data[0].cuda(0))
            test_label = Variable(data[1].cuda(0))
            sex_label = Variable(data[2].cuda(0))
            output, pred, _,_,_ = model(inputs)
            test_scores = pred.detach().argmax(dim=1)
            test_acc = (test_scores == test_label.argmax(dim=1)).float().sum().item()
            test_acc_total += test_acc
            total_num += len(test_label)
            loss+=loss_function(inputs,output)
            label_list.append(test_label.argmax(dim=1).cpu())
            pred_list.append(pred.argmax(dim=1).cpu())
            sex_list.append(sex_label.cpu())
        print('test_acc:', str(test_acc_total / total_num * 100))
    loss_mean=loss/(step+1)
    y_true = np.vstack(label_list)
    y_pred = np.vstack(pred_list)
    y_true = y_true.reshape(y_true.shape[0] * y_true.shape[1])
    y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1])
    C = confusion_matrix(y_true.tolist(), y_pred.tolist())

    y_sex = np.vstack(sex_list)
    female_indices = (y_sex == 0).squeeze()
    male_indices = (y_sex == 1).squeeze()
    # 分开后的预测标签和真实标签
    predicate_label_male = y_pred[male_indices]
    true_label_male = y_true[male_indices]

    predicate_label_female = y_pred[female_indices]
    true_label_female = y_true[female_indices]

    # 计算混淆矩阵
    confusion_matrix_male = confusion_matrix(true_label_male, predicate_label_male)
    confusion_matrix_female = confusion_matrix(true_label_female, predicate_label_female)
    return test_acc_total / total_num*100, C,confusion_matrix_male,confusion_matrix_female

def pretrain_model_gan(dataset_s, dataset_u, dataset_test, model, loss_fn, LR, EPOCH, domain,gender, gan ,norm_type, dataset_type, block, fold, r,tripledomain):
    opti = optim.Adam(model.parameters(), lr=LR)
    opti_loss = optim.Adam(loss_fn.parameters(), lr=LR / 5)
    loss_train = np.zeros(EPOCH)
    loss_valid = np.zeros(EPOCH)
    acc_list = np.zeros(EPOCH)
    best_test_acc = 0
    loss_function = torch.nn.MSELoss(reduction='mean')
    for epoch in range(EPOCH):
        running_loss_re = 0.0
        running_loss_dis = 0.0
        running_loss_ce = 0.0
        total_num, source_acc_total = 0, 0
        model.train()
        setup_seed(20)
        for step, (batch_s, batch_u, batch_test) in enumerate(zip(dataset_s, dataset_u, dataset_test)):
            s_data = Variable(batch_s[0].cuda(0))
            s_label = Variable(batch_s[1].cuda(0))
            s_label_sex = Variable(batch_s[2].cuda(0))

            u_data = Variable(batch_u[0].cuda(0))
            u_label = Variable(batch_u[1].cuda(0))
            u_label_sex = Variable(batch_u[2].cuda(0))

            t_data = Variable(batch_test[0].cuda(0))
            t_label = Variable(batch_test[1].cuda(0))
            t_label_sex = Variable(batch_test[2].cuda(0))

            inputs = torch.cat((s_data, u_data, t_data))
            label_sex = torch.cat((s_label_sex, u_label_sex, t_label_sex))
            loss_fn, opti_loss = train_discrinator(inputs, loss_fn, model, opti_loss)

            model.train()
            opti.zero_grad()
            outputs, pred, domain_loss, gender_loss, code = model(inputs, label_sex=label_sex)

            source_pred = pred[0:len(s_data), :]
            log_prob = torch.nn.functional.log_softmax(source_pred, dim=1)
            celoss = -torch.sum(log_prob * s_label) / len(s_label)

            loss_re = loss_function(inputs, outputs)
            loss_dis = 0.5 * loss_fn(inputs, outputs)

            # loss = loss_re - loss_dis + celoss + domain*domain_loss
            loss = gan*(loss_re - loss_dis) + celoss + domain * domain_loss + gender*gender_loss

            loss.backward()
            opti.step()

            running_loss_re += loss_re.data
            running_loss_dis += loss_dis.data
            running_loss_ce += celoss.data
        loss_train[epoch] = running_loss_re / (step + 1)
        source_scores = source_pred.detach().argmax(dim=1)
        source_acc = (source_scores == s_label.argmax(dim=1)).float().sum().item()
        source_acc_total += source_acc
        total_num += len(s_label)
        print('EPOCH_pre', str(epoch), 'Training Loss_re:', str(loss_train[epoch]))
        print('EPOCH_pre', str(epoch), 'Training Loss_dis:', str(running_loss_dis / (step + 1)))
        print('EPOCH_pre', str(epoch), 'Training Loss_ce:', str(running_loss_ce / (step + 1)))
        print('EPOCH_pre', str(epoch), 'train_acc:', str(source_acc_total / total_num * 100))
        target_test_acc, c,c_male,c_female = model_validation_autoencoder(model, dataset_test)
        acc_list[epoch] = target_test_acc
        if best_test_acc <= target_test_acc:
            best_test_acc = target_test_acc
            best_c_male = c_male
            best_c_female = c_female
            best_c = c
            best_epoch = epoch
            os.chdir(sys_path + '\\results')
            if tripledomain:
                torch.save(model.state_dict(),'autoencoder_gan_imp_384_{}_{}_block{}_tripledomain{}_gender{}_gan{}_fold{}_r{}_S0.75_semi.pkl'.format(norm_type, dataset_type, block, domain, gender, gan, fold, r))
            else:
                torch.save(model.state_dict(), 'autoencoder_gan_imp_384_{}_{}_block{}_domain{}_gender{}_gan{}_fold{}_r{}_S0.75_semi.pkl'.format(norm_type, dataset_type, block, domain, gender, gan, fold, r))
    print('best_acc:', best_test_acc)
    return best_test_acc, acc_list, best_c, best_epoch,best_c_male,best_c_female

def train_Autoencoder_gan(norm_type,dataset_type,finetune=0, block=1, domain = 1, gender = 1,gan=1,tripledomain=1):
    BATCH_SIZE = 128
    EPOCH = 80
    nflod = 10
    random_num = 100
    target_acc_curve = np.zeros((random_num, nflod, EPOCH))
    best_acc_mat = np.zeros((random_num, nflod))
    best_epoch_mat = np.zeros((random_num, nflod))
    best_c_mat = np.zeros((random_num, nflod, 2, 2))
    best_c_male_mat = np.zeros((random_num, nflod, 2, 2))
    best_c_female_mat = np.zeros((random_num, nflod, 2, 2))

    nNSSI_male_list =[]
    NSSI_male_list = []
    nNSSI_female_list = []
    NSSI_female_list = []
    NSSI_female_random_list = []
    nNSSI_male_random_list = []
    sexual_index = [0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	1,	1,	1,	0,	1,	0,	1,	1,
                    0,	1,	0,	1,	0,	1,	1,	1,	0,	0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,
                    0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,
                    0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,
                    0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	1,	0,	0]

    for i in range(114):
        if i<=36:
            if sexual_index[i] == 0:
                nNSSI_female_list.append(i)
            else:
                nNSSI_male_list.append(i)
        if i>36:
            if sexual_index[i] == 0:
                NSSI_female_list.append(i)
            else:
                NSSI_male_list.append(i)
    setup_seed(20)
    for i in range(random_num):
        NSSI_female_random = random.sample(NSSI_female_list, len(nNSSI_female_list))
        NSSI_female_random_list.append(NSSI_female_random)
        nNSSI_male_random = random.sample(nNSSI_male_list, len(NSSI_male_list))
        nNSSI_male_random_list.append(nNSSI_male_random)

    setup_seed(20)
    skf = KFold(n_splits=nflod, shuffle=True)
    for r in range(1):
        no_nssi_train_f = []
        no_nssi_test_f = []
        have_nssi_train_f = []
        have_nssi_test_f = []
        no_nssi_train_m = []
        no_nssi_test_m = []
        have_nssi_train_m = []
        have_nssi_test_m = []
        for fold, (a, b) in enumerate(skf.split(NSSI_female_random_list[r], nNSSI_female_list)):
            have_nssi_train_f.append(np.array(NSSI_female_random_list)[r][a])
            have_nssi_test_f.append(np.array(NSSI_female_random_list)[r][b])
            no_nssi_train_f.append(np.array(nNSSI_female_list)[a])
            no_nssi_test_f.append(np.array(nNSSI_female_list)[b])

        for fold, (a, b) in enumerate(skf.split(nNSSI_male_random_list[r], NSSI_male_list)):
            no_nssi_train_m.append(np.array(nNSSI_male_random_list)[r][a])
            no_nssi_test_m.append(np.array(nNSSI_male_random_list)[r][b])
            have_nssi_train_m.append(np.array(NSSI_male_list)[a])
            have_nssi_test_m.append(np.array(NSSI_male_list)[b])

        for fold in range(nflod):
            train_idx_f = np.hstack((no_nssi_train_f[fold], have_nssi_train_f[fold]))
            test_idx_f = np.hstack((no_nssi_test_f[fold], have_nssi_test_f[fold]))

            train_idx_m = np.hstack((no_nssi_train_m[fold], have_nssi_train_m[fold]))
            test_idx_m = np.hstack((no_nssi_test_m[fold], have_nssi_test_m[fold]))

            s_index_f = random.sample(train_idx_f.tolist(), int(len(train_idx_f) * 0.75))
            u_index_f = [i for num, i in enumerate(train_idx_f.tolist()) if i not in s_index_f]

            s_index_m = random.sample(train_idx_m.tolist(), int(len(train_idx_m) * 0.75))
            u_index_m = [i for num, i in enumerate(train_idx_m.tolist()) if i not in s_index_m]

            test_idx = np.hstack((test_idx_f, test_idx_m))
            s_index = np.hstack((s_index_f, s_index_m))
            u_index = np.hstack((u_index_f, u_index_m))

            s_data_set, u_data_set, test_data_set = get_dataset_nssi(norm_type, 384, block, s_index, u_index, test_idx, sexual_index)

            torch_dataset_s = Data.TensorDataset(torch.from_numpy(s_data_set['feature'].astype('float32')),
                                                 torch.from_numpy(s_data_set['label'].astype('float32')),
                                                 torch.from_numpy(s_data_set['label_sex'].astype('float32')))
            torch_dataset_u = Data.TensorDataset(torch.from_numpy(u_data_set['feature'].astype('float32')),
                                                 torch.from_numpy(u_data_set['label'].astype('float32')),
                                                 torch.from_numpy(u_data_set['label_sex'].astype('float32')))
            torch_dataset_valid = Data.TensorDataset(torch.from_numpy(test_data_set['feature'].astype('float32')),
                                                     torch.from_numpy(test_data_set['label'].astype('float32')),
                                                     torch.from_numpy(test_data_set['label_sex'].astype('float32')))

            setup_seed(20)
            model=Autoencoder_imp_nssi(16,1,1,384,tripledomain).cuda(0)

            loss_fn=Adversial_Loss_nssi(0.1,384).cuda(0)

            setup_seed(20)
            model.apply(weigth_init)
            loss_fn.apply(weigth_init)
            loader_s = Data.DataLoader(
                dataset=torch_dataset_s,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )
            loader_u = Data.DataLoader(
                dataset=torch_dataset_u,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )
            loader_valid = Data.DataLoader(
                dataset=torch_dataset_valid,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )
            best_test_acc, target_test_acc, best_c, best_epoch,best_c_male,best_c_female=pretrain_model_gan(loader_s, loader_u, loader_valid, model, loss_fn, 0.001, EPOCH, domain,gender,gan, norm_type, dataset_type, block, fold, r,tripledomain)
            target_acc_curve[r, fold, :] = target_test_acc
            best_acc_mat[r, fold] = best_test_acc
            best_c_mat[r, fold] = best_c
            best_c_male_mat[r, fold] = best_c_male
            best_c_female_mat[r, fold] = best_c_female
            best_epoch_mat[r, fold] = best_epoch

    os.chdir(sys_path + '\\results')
    result_list = {'best_acc_mat': best_acc_mat, 'target_acc_curve': target_acc_curve, 'no_nssi_train_f':no_nssi_train_f, 'no_nssi_test_f':no_nssi_test_f,
                   'have_nssi_train_f':have_nssi_train_f, 'have_nssi_test_f':have_nssi_test_f,'no_nssi_train_m':no_nssi_train_m, 'no_nssi_test_f':no_nssi_test_m,
                   'have_nssi_train_m':have_nssi_train_m, 'have_nssi_test_m':have_nssi_test_m, 'confusion_matrix':best_c_mat, 'best_epoch_mat':best_epoch_mat,'confusion_matrix_male':best_c_male_mat,'confusion_matrix_female':best_c_female_mat}
    if tripledomain:
        np.save('result_list_autoencoder_gan_imp_384_{}_{}_block{}_tripledomain{}_gender{}_gan{}_r{}_S0.75_semi.npy'.format(norm_type, dataset_type, block, domain,gender,gan,r), result_list)
        c = np.load('result_list_autoencoder_gan_imp_384_{}_{}_block{}_tripledomain{}_gender{}_gan{}_r{}_S0.75_semi.npy'.format(norm_type, dataset_type, block, domain,gender,gan,r), allow_pickle=True).item()
    else:
        np.save('result_list_autoencoder_gan_imp_384_{}_{}_block{}_domain{}_gender{}_gan{}_r{}_S0.75_semi.npy'.format(norm_type, dataset_type, block, domain,gender,gan,r), result_list)
        c = np.load('result_list_autoencoder_gan_imp_384_{}_{}_block{}_domain{}_gender{}_gan{}_r{}_S0.75_semi.npy'.format(norm_type, dataset_type, block, domain,gender,gan,r), allow_pickle=True).item()
    return c
c = train_Autoencoder_gan('global_gaussian_value', 'nssi', block=1, domain = 1, gender = 1, gan=1, tripledomain = 1)
best_acc_mat = c['best_acc_mat']
c_mean = np.mean(c['best_acc_mat'])
c_std = np.std(c['best_acc_mat'])





