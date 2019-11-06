from __future__ import print_function, division

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
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

    def __load_frames(self, paths):
        """
        NOTE THAT WE NEED DIFFERENT SIZE FRAME AT  DIFFERENT PYRAMID LEVEL SO WE RESIZE HERE
        WE ARE GOING TO USE RANDOM CROP TECHNIQUES
        """

        images = [*map(lambda x: Image.open(x), paths)]
        final_frame = torch.cat([ToTensor()(x) for x in images], 0)

        if not self.is_test:
            pyra1_frame = torch.cat([ToTensor()(x.resize((256, 108))) for x in images], 0)
            pyra2_frame = torch.cat([ToTensor()(x.resize((128, 54))) for x in images], 0)
            laten_frame = torch.cat([ToTensor()(x.resize((64, 27))) for x in images], 0)

            return {'frame': final_frame,
                    'pyra1_frame': pyra1_frame,
                    'pyra2_frame': pyra2_frame,
                    'laten_frame': laten_frame}
        else:
            return {'frame': final_frame}

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
                 batch_sizes=(20, 10, 1), num_workers=8):
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