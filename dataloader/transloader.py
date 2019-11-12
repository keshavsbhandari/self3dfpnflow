from __future__ import print_function, division

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
import torch.nn.functional as F

from torchvision.transforms import ToPILImage, ToTensor, Resize

from PIL import Image
from utils import (get_train_val_test_list, getflow, ResizeImage)

USE_CUT_OFF = 10
import random

"""
torch.Size([1, 2, 9, 436, 1024]) torch.Size([1, 2, 9, 436, 1024])
torch.Size([1, 2, 9, 108, 256]) torch.Size([1, 2, 9, 108, 256])
torch.Size([1, 2, 9, 54, 128]) torch.Size([1, 2, 9, 54, 128])
torch.Size([1, 2, 9, 27, 64]) torch.Size([1, 2, 9, 27, 64])
"""


class SintelDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, is_test=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = dataset
        self.is_test = is_test
        self.transform = transform
        self.unfold = torch.nn.Unfold(kernel_size=3, padding=1)

    def __len__(self):
        return len(self.dataset)

    def transEstimateFlow(self, pic1, pic2, newsize=(128, 54)):
        # note optimum newsize is (256,108
        def getnearindex(dmv, x):
            sqdiff = torch.sum(torch.pow((x - dmv).float(), 2), 1)
            sssq = torch.pow(sqdiff, 0.5)
            return sssq.min(0)[1]

        def getslice(dmv, x, y):
            val = (dmv == x).nonzero()
            if not val.nelement():
                return y
            else:
                indx = getnearindex((dmv == x).nonzero(), y)
                return val[indx]

        maxsum = 16777215.0
        ar1 = np.array(pic1.resize(newsize, 3))
        ar2 = np.array(pic2.resize(newsize, 3))

        tobit = lambda x: bin(x).replace('0b', '').zfill(8)
        trans = lambda tmp: f"0b{''.join([*map(lambda x: tobit(x), tmp)])}"

        out1 = np.apply_along_axis(lambda x: int(trans(x), 2), 2, ar1).astype(np.float32)
        out2 = np.apply_along_axis(lambda x: int(trans(x), 2), 2, ar2).astype(np.float32)

        ten1 = torch.tensor(out1) / maxsum
        ten2 = torch.tensor(out2) / maxsum

        # ten1 = ten1.cuda()#CUda
        # ten2 = ten2.cuda()#CUda

        indi = (ten2 == ten2).nonzero()  # .cuda()#CUDA
        zipped = zip(ten2.flatten(), indi)
        forward = torch.stack([*map(lambda x: getslice(ten1, x[0], x[1]), zipped)])
        forward = indi - forward
        forward = forward.view(*ten1.shape, -1).permute(2, 0, 1)
        forward = forward.float() / torch.tensor([*newsize[::-1]]).view(2, 1, 1).float()

        zipped = zip(ten1.flatten(), indi)
        backward = torch.stack([*map(lambda x: getslice(ten2, x[0], x[1]), zipped)])
        backward = indi - backward
        backward = backward.view(*ten2.shape, -1).permute(2, 0, 1)
        backward = backward.float() / torch.tensor([*newsize[::-1]]).view(2, 1, 1).float()
        return forward, backward

    def __load_frames(self, paths):
        """
        NOTE THAT WE NEED DIFFERENT SIZE FRAME AT  DIFFERENT PYRAMID LEVEL SO WE RESIZE HERE
        WE ARE GOING TO USE RANDOM CROP TECHNIQUES
        """

        images = [*map(lambda x: Image.open(x), paths)]

        forward, backward = self.transEstimateFlow(*images)

        final_frame = torch.cat([ToTensor()(x) for x in images], 0)

        return {'frame': final_frame,
                'ff': forward,
                'fb': backward}

        # if not self.is_test:
        #     pyra1_frame = torch.cat([ToTensor()(x.resize((256, 108))) for x in images], 0)
        #     pyra2_frame = torch.cat([ToTensor()(x.resize((128, 54))) for x in images], 0)
        #     laten_frame = torch.cat([ToTensor()(x.resize((64, 27))) for x in images], 0)
        #
        #     return {'frame': final_frame,
        #             'pyra1_frame': pyra1_frame,
        #             'pyra2_frame': pyra2_frame,
        #             'laten_frame': laten_frame,
        #             'ff': forward,
        #             'fb': backward,}
        # else:
        #     return {'frame': final_frame,
        #             'ff': forward,
        #             'fb': backward}

    def __load_occlusion(self, paths):
        return ToTensor()(Image.open(paths[0]))

    def __load_flow(self, flow_path):
        flow_numpy = getflow(flow_path[0])
        flowtorch = torch.tensor(flow_numpy).permute(2, 0, 1)
        flowtorch = flowtorch / torch.tensor([436., 1024.]).view(2, 1, 1)
        return flowtorch

    def __getitem__(self, idx):
        instances = self.dataset[idx]
        sample = self.__load_frames(instances['frame'])
        if not self.is_test:
            sample['occlusion'] = self.__load_occlusion(instances['occlusion'])
            sample['flow'] = self.__load_flow(instances['flow'])
        return sample


class SintelLoader(object):
    def __init__(self, train_path="/data/keshav/sintel/training/final", test_path="/data/keshav/sintel/test/final",
                 batch_sizes=(10, 10, 1), num_workers=8):
        self.trainset, self.valset, self.testset = get_train_val_test_list(trainpath=train_path, testpath=test_path,
                                                                           n=2)

        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        self.load_all()

    def load_all(self):
        random.shuffle(self.trainset)
        random.shuffle(self.valset)
        random.shuffle(self.testset)

        self.__train = SintelDataset(dataset=self.trainset)
        self.__val = SintelDataset(dataset=self.valset)
        self.__test = SintelDataset(dataset=self.testset, is_test=True)

        self.trainloader = DataLoader(self.__train, batch_size=self.batch_sizes[0], num_workers=self.num_workers,
                                      pin_memory=True, )
        self.valloader = DataLoader(self.__val, batch_size=self.batch_sizes[1], num_workers=self.num_workers,
                                    pin_memory=True)

        self.testloader = DataLoader(self.__test, batch_size=self.batch_sizes[2], num_workers=self.num_workers,
                                     pin_memory=True)

    def train(self):
        self.load_all()
        return self.trainloader

    def val(self):
        self.load_all()
        return self.valloader

    def test(self):
        self.load_all()
        return self.testloader

    def train_val_test(self):
        self.load_all()
        return self.trainloader, self.valloader, self.testloader
