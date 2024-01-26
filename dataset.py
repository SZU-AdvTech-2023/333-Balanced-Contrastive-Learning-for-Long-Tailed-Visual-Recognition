import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from collections import Counter
from PIL import Image

def UT_HAR_dataset(root_dir):
    data_list = glob.glob(root_dir+'/UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'/UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data


# dataset: /class_name/xx.mat
class CSI_Dataset(Dataset):
    """CSI dataset."""
    #cls_num = 9
    def __init__(self, root_dir, modal='CSIamp', transform=None,  cls_num=9, target_transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_dict = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            8: 7,
            9: 8

        }
        self.cls_num = cls_num
        self.root_dir = root_dir
        self.modal=modal
        self.transform = transform
        self.data_list = glob.glob(root_dir+'/*/*.mat')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}
        print('category:', self.category)

    def __len__(self):
        return len(self.data_list)


    def get_cls_num_list(self):
        labels = [self.extract_label(data_path) for data_path in self.data_list]
        self.targets=labels
        label_counts = Counter(labels)
        self.num_per_cls_dict = dict(label_counts)
        cls_num_list = []
        for i in range(1, self.cls_num+1):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



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
        data_path = self.data_list[index]  # src

        with h5py.File(data_path, 'r') as f1:
            amp1 = f1['csi_amp']
            fea = amp1[()]
            # if fea.shape == (480, 30, 3):
            # fea = np.swapaxes(fea, 0, 2)

        fea = np.float32(fea)
        # Assuming you want to select the first channel
        # img_data = fea[:, :, 0]
        img_data = fea
        # Normalize the data to the range [0, 255] if needed
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255.0

        # Convert to uint8 and create a PIL Image
        img_data = img_data.astype(np.uint8)
        img = Image.fromarray(img_data)

        extract_label = self.extract_label(data_path)
        target = self.label_dict[extract_label]
        self.targets = target

        if self.transform is not None:
            sample1 = self.transform[0](img)
            sample2 = self.transform[1](img)
            sample3 = self.transform[2](img)

        # if self.target_transform is not None:
        #     target = self.target_transform(label)

        return [sample1, sample2, sample3], target


class CSI_Dataset_val(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_dict = {
            1: 0,
            2: 1,
            10: 2,
            11: 3

        }
        self.root_dir = root_dir
        self.modal = modal
        self.transform = transform
        self.data_list = glob.glob(root_dir + '/*/*.mat')
        self.folder = glob.glob(root_dir + '/*/')
        self.category = {self.folder[i].split('/')[-2]: i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

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

    def __getitem__(self, idx):
        data_path = self.data_list[idx]  # src

        with h5py.File(data_path, 'r') as f1:
            amp1 = f1['csi_amp']
            fea = amp1[()]


        fea = np.float32(fea)
        # Assuming you want to select the first channel
        # img_data = fea[:, :, 0]
        img_data = fea
        # Normalize the data to the range [0, 255] if needed
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255.0

        # Convert to uint8 and create a PIL Image
        img_data = img_data.astype(np.uint8)
        img = Image.fromarray(img_data)

        extract_label = self.extract_label(data_path)
        label = self.label_dict[extract_label]

        if self.transform is not None:
            sample = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(label)

        return sample, label


class Widar_Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir+'/*/*.csv')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = np.genfromtxt(sample_dir, delimiter=',')
        
        # normalize
        x = (x - 0.0025)/0.0119
        
        # reshape: 22,400 -> 22,20,20
        x = x.reshape(22,20,20)
        # interpolate from 20x20 to 32x32
        # x = self.reshape(x)
        x = torch.FloatTensor(x)

        return x,y

