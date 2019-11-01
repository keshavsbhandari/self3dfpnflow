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

    def frame_format(self, paths, hs, he, ws, we):
        """
        NOTE THAT WE NEED DIFFERENT SIZE FRAME AT  DIFFERENT PYRAMID LEVEL SO WE RESIZE HERE
        WE ARE GOING TO USE RANDOM CROP TECHNIQUES
        """
        t0 = transforms.Compose([ToTensor()])
        t1 = transforms.Compose([Resize((64, 64)), ToTensor()])
        t2 = transforms.Compose([Resize((32, 32)), ToTensor()])
        t3 = transforms.Compose([Resize((16, 16)), ToTensor()])

        images = [*map(lambda x:Image.open(x),paths)]

        R = lambda x, t: t(x).unsqueeze(1)[:,:,hs:he, ws:we]

        C = lambda x:ToPILImage()((ToTensor()(x))[:,hs:he, ws:we])

        final_frame = torch.cat([*map(lambda img: R(img, t0),images)], 1)

        if not self.is_test:

            f1 = []
            f2 = []
            f3 = []

            for img in images:
                tenim = ToTensor()(img)
                tenim = tenim[:,hs:he, ws:we]
                tenim = ToPILImage()(tenim)
                f1im = ToTensor()(tenim.resize((64,64)))
                f2im = ToTensor()(tenim.resize((32,32)))
                f3im = ToTensor()(tenim.resize((16,16)))

                f1.append(f1im.unsqueeze(1))
                f2.append(f2im.unsqueeze(1))
                f3.append(f3im.unsqueeze(1))


            pyra1_frame = torch.cat(f1, 1)
            pyra2_frame = torch.cat(f2, 1)
            laten_frame = torch.cat(f3, 1)

            return final_frame, pyra1_frame, pyra2_frame, laten_frame
        else:
            return final_frame



    def get_random_crop(self):
        hs = random.randint(0, 436 - 260)
        he = hs + 260
        ws = random.randint(0, 1024 - 256)
        we = ws + 256

        return hs, he, ws, we

    def __load_frames(self, paths, hs, he, ws, we):
        if not self.is_test:
            final_frame, pyra1_frame, pyra2_frame, laten_frame = self.frame_format(paths, hs, he, ws, we)
            return final_frame, pyra1_frame, pyra2_frame, laten_frame
        else:
            final_frame = self.frame_format(paths, hs, he, ws, we)
            return final_frame


    def __load_occlusion(self, paths, hs, he, ws, we):
        t0 = transforms.Compose([ToTensor()])
        R = lambda x, t: t(x).unsqueeze(1)[:, :, hs:he, ws:we]
        occlist = [*map(lambda x: R(Image.open(x), t0), paths)]
        return torch.cat(occlist,1)

    def flow_format(self, flow_path):
        flow_numpy = getflow(flow_path)
        flowtorch = torch.tensor(flow_numpy).permute(2, 0, 1).unsqueeze(1)
        # flowtorch = flowtorch / torch.tensor([436., 1024.]).view(2, 1, 1, 1)
        return flowtorch

    def __load_flow(self, paths):
        return torch.cat([*map(lambda x: self.flow_format(x), paths)], 1)

    def __getitem__(self, idx):
        hs, he, ws, we = self.get_random_crop()
        instances = self.dataset[idx]
        sample = {}
        if not self.is_test:
            sample['frame'], sample['pyra1_frame'], sample['pyra2_frame'], sample['laten_frame'] = self.__load_frames(
                instances['frame'], hs, he, ws, we)
        else:
            sample['frame'] = self.__load_frames(instances['frame'], hs, he, ws, we)

        if not self.is_test:
            sample['occlusion'] = self.__load_occlusion(instances['occlusion'], hs, he, ws, we)
            sample['flow'] = self.__load_flow(instances['flow'])[:, :, hs:he, ws:we]
        return sample


class SintelLoader3D(object):
    def __init__(self, train_path="/data/keshav/sintel/training/final", test_path="/data/keshav/sintel/test/final",
                 batch_sizes=(5, 1, 1), num_workers=8):
        self.trainset, self.valset, self.testset = get_train_val_test_list(train_path, test_path)

        self.batch_sizes = batch_sizes
        self.num_workers = num_workers

        self.__train = SintelDataset3D(dataset=self.trainset)
        self.__val = SintelDataset3D(dataset=self.valset)
        self.__test = SintelDataset3D(dataset=self.testset, is_test=True)

        self.load_all()

    def load_all(self):
        self.trainloader = DataLoader(self.__train, batch_size=self.batch_sizes[0], num_workers=self.num_workers,
                                      pin_memory=True,)
        self.valloader = DataLoader(self.__val, batch_size=self.batch_sizes[1], num_workers=self.num_workers,
                                    pin_memory=True)

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
