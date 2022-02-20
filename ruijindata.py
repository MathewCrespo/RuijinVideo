from PIL import Image
import torch
import os
import sys
from torch.utils.data import Dataset
from torchvision import transforms
import xml.etree.cElementTree as ET
from tqdm import tqdm
import random
import pandas as pd
from random import randint,sample
import av
import numpy as np
# import cv2
# import selectivesearch

class RuijinData(Dataset):
    def __init__(self, root, pre_transform = None, sub_list = [0,1,2,3,4], task = 'BM', modality = 'image'):
        self.root = root
        self.task = task
        self.pre_transform = pre_transform
        self.sub_list = sub_list
        self.label_table = pd.read_csv(os.path.join(self.root, 'final_label.csv'))
        self.bm_patient_info = []
        self.alnm_patient_info = []
        self.modality = modality
        if self.modality not in ['video', 'image']:
            raise NotImplementedError("Modality Error")
        if self.task not in ['BM', 'ALNM']:
            raise NotImplementedError('Task Error')
        for fold in self.sub_list:
            self.scan(fold)
            

    def scan(self, fold):
        fold_table = self.label_table[self.label_table['{}_fold'.format(self.task)]==fold].reset_index(drop=True)
        for k in range(len(fold_table)):
            id = fold_table.loc[k,'ID']
            p_path = os.path.join(self.root,str(id).zfill(9))
            p_label = fold_table.loc[k,self.task]
            now_patient = {}
            now_patient['label'] = p_label
            filesOfPatient = os.listdir(p_path)
            now_patient['video_root'] = [os.path.join(p_path, x) for x in filesOfPatient if x.endswith('.mp4')]
            now_patient['img_root'] = [os.path.join(p_path, x) for x in filesOfPatient if x.endswith('.JPG') or x.endswith('.jpg')]  # .jpg should also be read
            if self.task =='BM':
                self.bm_patient_info.append(now_patient)
            else:
                self.alnm_patient_info.append(now_patient)
    
    def __getitem__(self, index):
        now_patient = self.bm_patient_info[index] if self.task=='BM' else self.alnm_patient_info[index]
        label = now_patient['label']
        label = torch.tensor(label)  # tensor type

        if self.modality == 'image':
            imgs = []
            for img_path in now_patient['img_root']:
                img = Image.open(img_path).convert('RGB')
                if self.pre_transform is not None:
                    img = self.pre_transform(img)
                imgs.append(img)
            return torch.stack([x for x in imgs], dim=0), label
        else:
            video = self.get_tensor_from_video(now_patient['video_root'][0])
            return video, label
    
    def get_tensor_from_video(self, video_path,is_multi_thread_decode = True):
        """
        :param video_path: 视频文件地址
        :param is_multi_thread_decode: 是否多线程解码文件
        :return: pytorch tensor
        """
        if not os.access(video_path, os.F_OK):
            print('测试文件不存在')
            return

        container = av.open(video_path)
        if is_multi_thread_decode:
            container.streams.video[0].thread_type = "AUTO"

        container.seek(0, any_frame=False, backward=True, stream=container.streams.video[0])

        frames = []
        for frame in container.decode(video=0):
            frames.append(frame)
        container.close()

        # result_frams = None

        # 从视频帧转换为ndarray
        result_frames = [frame.to_rgb().to_ndarray() for frame in frames]
        # 转换成tensor
        result_frames = np.stack(result_frames)
        # 注意：此时result_frames组成的维度为[视频帧数量，高，宽，通道数]


        if self.pre_transform is not None:
            out_frames = torch.stack((self.pre_transform(result_frames[0]),))
            for i in range(1, result_frames.shape[0]):
                frame = self.pre_transform(result_frames[i])
                out_frames = torch.cat((out_frames, frame.unsqueeze(0)))
        return out_frames


    def __len__(self):
        if self.task == 'BM':
            return len(self.bm_patient_info)
        else:
            return len(self.alnm_patient_info)

if __name__ == '__main__':
    pre_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    ])
    dataset = RuijinData(root = '/remote-home/share/RJ_video/RJ_video_crop', pre_transform = pre_transform, task = 'ALNM', modality = 'image')
    print(dataset[1][0].shape)
    print(len(dataset))