#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Copyright (c) 2019 gyfastas
'''
from __future__ import absolute_import
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import utils.utility as utility
from utils.logger import Logger
import argparse
from importlib import import_module
from utils.logger import Logger
from torch.optim import Adam
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from ruijindata import RuijinData
from models.VideoModel import Res_Attention
from Trainer.ImageTrainer import ImageTrainer




class Image_config(object):
    def __init__(self, log_root, args):
        #self.net = getattr(import_module('models.graph_attention'),args.net)(t=args.t, task=args.task)
        #print(self.net)
        self.net = Res_Attention()
        self.net = self.net.cuda()
        self.train_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    #transforms.ColorJitter(brightness = 0.25),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    # transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
                    transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
        ])
        self.optimizer = Adam(self.net.parameters(), lr=args.lr)
        self.lrsch = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 30, 50, 70], gamma=0.5)
        self.logger = Logger(log_root)
        self.trainbag = RuijinData(args.data_root, pre_transform = self.train_transform, sub_list=[x for x in [0,1,2,3,4] if x!=args.test_fold], 
                                    task = args.task, modality = args.modality)
        print(len(self.trainbag))
        self.testbag = RuijinData(args.data_root, pre_transform = self.test_transform, sub_list = [args.test_fold], 
                                    task = args.task, modality = args.modality)
        self.train_loader = DataLoader(self.trainbag, batch_size=args.batchsize, shuffle=True, num_workers=8)
        self.val_loader = DataLoader(self.testbag, batch_size=args.batchsize, shuffle=False, num_workers=8)
        self.trainer = ImageTrainer(self.net, self.optimizer, self.lrsch, None, self.train_loader, self.val_loader, self.logger, 0)
        self.save_config(args)

    def save_config(self,args):
        config_file = './saved_configs/'+args.log_root+'.txt'
        f = open(config_file,'a+')
        argDict = args.__dict__
        for arg_key, arg_value in argDict.items():
            f.writelines(arg_key+':'+str(arg_value)+'\n')
        f.close()
        self.logger.auto_backup('./')
        self.logger.backup_files([config_file])




if __name__=='__main__':
    #configs = getattr(import_module('configs.'+args.config),'Config')()
    #configs = configs.__dict__
    parser = argparse.ArgumentParser(description='Ruijin Framework')
    parser.add_argument('--data_root',type=str,default='/remote-home/share/RJ_video_crop')
    parser.add_argument('--log_root',type=str)
    parser.add_argument('--test_fold',type=int,default=0, help='which fold of data is used for test')
    parser.add_argument('--task',type=str,default='BM', help='BM or ALNM')
    parser.add_argument('--modality', type=str, help='Image or Video')
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--epoch',type=int,default=50)
    parser.add_argument('--resume',type=int,default=-1)
    parser.add_argument('--batchsize',type=int,default=1)
    parser.add_argument('--net',type=str,default='H_Attention_Graph')

    # parse parameters
    args = parser.parse_args()
    log_root = os.path.join('/remote-home/share/RJ_video/exps',args.log_root)
    if not os.path.exists(log_root):
        os.mkdir(log_root)

    
    config_object = Image_config(log_root,args)
    

    # train and eval
    for epoch in range(config_object.logger.global_step, args.epoch):
        print('Now epoch {}'.format(epoch))
        config_object.trainer.train()
        config_object.trainer.test()