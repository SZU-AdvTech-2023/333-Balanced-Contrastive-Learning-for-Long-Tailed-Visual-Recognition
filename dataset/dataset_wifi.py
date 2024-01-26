import pickle
import numpy as np
import os
import random

import h5py
import torch
from torch.utils.data import Dataset
#import mfsan_config as args


# seed = 2021
# os.environ["PYTHONHASHSEED"] = str(seed)
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# rng = np.random.default_rng(seed)
# generator = torch.Generator().manual_seed(seed)


class WiFiDataset(Dataset):
    def __init__(self, data_dir):
        with open(data_dir) as f:
            self.data = pickle.load(f)
            
        self.label_dict = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
        }

    def __len__(self):
        return len(self.data)

    @staticmethod
    def extract_label(datapath):  # ../user11/user11-1-1-r1-t6.mat
        filename = datapath.split('/')[-1]  # user11-1-1-r1-t6.mat
        filename = filename.split('.')[0]  # user11-1-1-r1-t6
        userid = filename.split('-')[0]
        userid = int(userid[4:])
        # trkid = filename.split('-')[1]
        # rxid = filename.split('-')[3][-1]
        # return [userid, trkid, rxid]  
        return userid  
    
    def __getitem__(self, index):
        data_path = self.data[index]  # src

        with h5py.File(data_path, 'r') as f1:
            amp1 = f1['csi_amp']
            fea = amp1[()]
        # if fea.shape == (480, 30, 3):
            fea = np.swapaxes(fea, 0, 2)

        fea = np.float32(fea)
        # horse_fea = norm(horse_fea)
        # label = self.extract_label(data_path)[0] - 1
        extract_label = self.extract_label(data_path)
        target = self.label_dict[extract_label]

        return fea, target
    


class AugOnceWiFiDataset(Dataset):
    def __init__(self, data_dir):
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)
        
        self.label_dict = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
        }
            
    def __len__(self):
        return len(self.data)

    @staticmethod
    def extract_label(datapath):  # ../user11/user11-1-1-r1-t6.mat
        filename = datapath.split('/')[-1]  # user11-1-1-r1-t6.mat
        filename = filename.split('.')[0]  # user11-1-1-r1-t6
        userid = filename.split('-')[0]
        userid = int(userid[4:])
        # trkid = filename.split('-')[1]
        # rxid = filename.split('-')[3][-1]
        # return [userid, trkid, rxid]  
        return userid  
    
    @staticmethod
    def strong_aug(data):
        data_aug = np.zeros_like(data)  # 存放数据

        time_shift_rate = random.choice(range(0, args.strong_time_shift_rate + 1))  # 随机挑time shift rate [0~cfg_time_shift_rate]
        # dropout_rate = random.choice(range(0, args.strong_dropout_rate + 1))   

        random_number = random.random()  # [0 - 1]  
        if random_number < 0.5:
            shift_rate = 1 - time_shift_rate / 100
        else:
            shift_rate = 1 + time_shift_rate / 100
        
        antenna, channel, period = data.shape
        select_period = round(period * shift_rate)

        for ant in range(antenna):
            for cha in range(channel):
                fp = data[ant, cha, :]
                xp = np.arange(period)
                xvals = np.linspace(0, period, num=select_period)
                yinterp = np.interp(xvals, xp, fp)
                if shift_rate > 1:
                    data_aug[ant, cha, :] = yinterp[:period]
                elif shift_rate < 1:
                    left = period - select_period
                    data_aug[ant, cha, :] = np.hstack((yinterp, yinterp[:left]))
                else:   # shift_rate == 1
                    pass
                # if dropout_rate > 0:
                #     dropout_num = int(dropout_rate / 100 * period)
                #     indices_rdn = random.sample(range(period), dropout_num)
                #     for indice in indices_rdn:                
                #         data_aug[ant, cha, indice] = 0        
        return data_aug
    
    def __getitem__(self, index):
        data_path = self.data[index]  # src

        with h5py.File(data_path, 'r') as f1:
            amp1 = f1['csi_amp']
            fea = amp1[()]
        # if fea.shape == (480, 30, 3):
            fea = np.swapaxes(fea, 0, 2)

        fea = np.float32(fea)
        fea_aug = self.strong_aug(fea)
        # label = self.extract_label(data_path)[0] - 1
        extract_label = self.extract_label(data_path)
        label = self.label_dict[extract_label]

        return fea, fea_aug, label        




class AugTwiceWiFiDataset(Dataset):
    def __init__(self, data_dir):
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)
        
        self.label_dict = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
        }
            
    def __len__(self):
        return len(self.data)

    @staticmethod
    def extract_label(datapath):  # ../user11/user11-1-1-r1-t6.mat
        filename = datapath.split('/')[-1]  # user11-1-1-r1-t6.mat
        filename = filename.split('.')[0]  # user11-1-1-r1-t6
        userid = filename.split('-')[0]
        userid = int(userid[4:])
        # trkid = filename.split('-')[1]
        # rxid = filename.split('-')[3][-1]
        # return [userid, trkid, rxid]  
        return userid  
    
    @staticmethod
    def strong_aug(data):
        data_aug = np.zeros_like(data)  # 存放数据

        time_shift_rate = random.choice(range(0, args.strong_time_shift_rate + 1))  # 随机挑time shift rate [0~cfg_time_shift_rate] 20-30左右
        # dropout_rate = random.choice(range(0, args.strong_dropout_rate + 1))   

        random_number = random.random()  # [0 - 1]  
        if random_number < 0.5:
            shift_rate = 1 - time_shift_rate / 100
        else:
            shift_rate = 1 + time_shift_rate / 100
        
        antenna, channel, period = data.shape
        select_period = round(period * shift_rate)

        for ant in range(antenna):
            for cha in range(channel):
                fp = data[ant, cha, :]
                xp = np.arange(period)
                xvals = np.linspace(0, period, num=select_period)
                yinterp = np.interp(xvals, xp, fp)
                if shift_rate > 1:
                    data_aug[ant, cha, :] = yinterp[:period]
                elif shift_rate < 1:
                    left = period - select_period
                    data_aug[ant, cha, :] = np.hstack((yinterp, yinterp[:left]))
                else:   # shift_rate == 1
                    pass
                # if dropout_rate > 0:
                #     dropout_num = int(dropout_rate / 100 * period)
                #     indices_rdn = random.sample(range(period), dropout_num)
                #     for indice in indices_rdn:                
                #         data_aug[ant, cha, indice] = 0        
        return data_aug


    @staticmethod
    def weak_aug(data):
        data_aug = np.zeros_like(data)  # 存放数据

        time_shift_rate = random.choice(range(0, args.weak_time_shift_rate + 1))  # 随机挑time shift rate [0~cfg_time_shift_rate] 5左右
        # dropout_rate = random.choice(range(0, args.strong_dropout_rate + 1))   

        random_number = random.random()  # [0 - 1]  
        if random_number < 0.5:
            shift_rate = 1 - time_shift_rate / 100
        else:
            shift_rate = 1 + time_shift_rate / 100
        
        antenna, channel, period = data.shape
        select_period = round(period * shift_rate)

        for ant in range(antenna):
            for cha in range(channel):
                fp = data[ant, cha, :]
                xp = np.arange(period)
                xvals = np.linspace(0, period, num=select_period)
                yinterp = np.interp(xvals, xp, fp)
                if shift_rate > 1:
                    data_aug[ant, cha, :] = yinterp[:period]
                elif shift_rate < 1:
                    left = period - select_period
                    data_aug[ant, cha, :] = np.hstack((yinterp, yinterp[:left]))
                else:   # shift_rate == 1
                    pass
                # if dropout_rate > 0:
                #     dropout_num = int(dropout_rate / 100 * period)
                #     indices_rdn = random.sample(range(period), dropout_num)
                #     for indice in indices_rdn:                
                #         data_aug[ant, cha, indice] = 0        
        return data_aug
    

    def __getitem__(self, index):
        data_path = self.data[index]  # src

        with h5py.File(data_path, 'r') as f1:
            amp1 = f1['csi_amp']
            fea = amp1[()]
        # if fea.shape == (480, 30, 3):
            fea = np.swapaxes(fea, 0, 2)

        fea = np.float32(fea)
        fea_aug = self.strong_aug(fea)
        fea_aug1 = self.strong_aug(fea)

        fea_weak_aug = self.weak_aug(fea)

        # label = self.extract_label(data_path)[0] - 1
        extract_label = self.extract_label(data_path)
        label = self.label_dict[extract_label]

        return fea, fea_aug, fea_aug1, fea_weak_aug, label        
        # return fea, fea_aug, fea_aug1, label        






