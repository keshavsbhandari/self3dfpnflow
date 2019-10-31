from __future__ import print_function, division

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from torchvision.transforms import ToPILImage, ToTensor, Resize

from PIL import Image
from utils import (get_train_val_test_list, getflow, ResizeImage)

USE_CUT_OFF = 10
import random


class SintelDataset3D(Dataset):
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

    def cst(self, frames):
        size = frames.size()
        frames = frames.permute(1, 0, 2, 3)  # as f is CDHW change to DCHW
        D, C, H, W = frames.size()
        img = self.unfold(frames).view(D, C, 9, -1).permute(0, 1, 3, 2)
        mid = img[:, :, :, 4].unsqueeze(-1)
        encoding = (img >= mid).float().view(-1, 9) * torch.tensor([128., 64., 32., 16., 0., 8., 4., 2., 1.])
        encoding = encoding.sum(-1).view(*size) / 255.
        return encoding

    def frame_format(self, paths, transform = None):
        img = Image.open(paths)
        if transform:
            img = transform(img).unsqueeze(1)
        else:
            img = ToTensor()(img).unsqueeze(1)

        return img

    def __load_frames(self, paths):

        final_frame = torch.cat([*map(lambda x: self.frame_format(x), paths)], 1)

        if not self.is_test:
            """
            NOTE THAT WE NEED DIFFERENT SIZE FRAME AT  DIFFERENT PYRAMID LEVEL SO WE RESIZE HERE
            """
            transform1 = transforms.Compose([Resize((108, 256)), ToTensor()])
            transform2 = transforms.Compose([Resize((54, 128)), ToTensor()])
            transform3 = transforms.Compose([Resize((27, 64)), ToTensor()])

            pyra1_frame = torch.cat([*map(lambda x: self.frame_format(x, transform1), paths)], 1)
            pyra2_frame = torch.cat([*map(lambda x: self.frame_format(x, transform2), paths)], 1)
            laten_frame = torch.cat([*map(lambda x: self.frame_format(x, transform3), paths)], 1)
            return final_frame, pyra1_frame, pyra2_frame, laten_frame
        else:
            return final_frame, None, None, None

    def flow_format(self, flow_path):
        flow_numpy = getflow(flow_path)
        flowtorch = torch.tensor(flow_numpy).permute(2, 0, 1).unsqueeze(1)
        flowtorch = flowtorch / torch.tensor([436., 1024.]).view(2, 1, 1, 1)
        return flowtorch

    def __load_flow(self, paths):
        return torch.cat([*map(lambda x: self.flow_format(x), paths)], 1)

    def __getitem__(self, idx):
        instances = self.dataset[idx]
        sample = {}

        sample['frame'], sample['pyra1_frame'], sample['pyra2_frame'], sample['laten_frame'] = self.__load_frames(instances['frame'])

        if not self.is_test:
            sample['occlusion'] = self.__load_frames(instances['occlusion'])
            sample['flow'] = self.__load_flow(instances['flow'])

        return sample


class SintelLoader3D(object):
    def __init__(self, train_path="/data/keshav/sintel/training/final", test_path="/data/keshav/sintel/test/final",
                 batch_sizes=(1, 1, 1), num_workers=8):
        self.trainset, self.valset, self.testset = get_train_val_test_list(train_path, test_path)

        self.batch_sizes = batch_sizes
        self.num_workers = num_workers

        self.__train = SintelDataset3D(dataset=self.trainset)
        self.__val = SintelDataset3D(dataset=self.valset)
        self.__test = SintelDataset3D(dataset=self.testset, is_test=True)
        self.load_all()

    def load_all(self):
        self.trainloader = DataLoader(self.__train, batch_size=self.batch_sizes[0], num_workers=self.num_workers,
                                 pin_memory=True)
        self.valloader = DataLoader(self.__val, batch_size=self.batch_sizes[1], num_workers=self.num_workers, pin_memory=True)

        self.testloader = DataLoader(self.__test, batch_size=self.batch_sizes[2], num_workers=self.num_workers,
                                pin_memory=True)

    def train(self):
        return self.trainloader

    def val(self):
        return self.valloader

    def test(self):
        return self.testloader

    def train_val_test(self):
        return self.trainloader, self.valloader, self.testloader