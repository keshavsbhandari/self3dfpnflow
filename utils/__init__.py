import numpy as np
import torch.nn.functional as F
import torch
from pathlib import Path
from itertools import chain
import re
import random
from torchvision.transforms import ToPILImage, ToTensor
import matplotlib.pyplot as plt

from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def getImGradient(imten, channel = 3, nocuda = False):
    Gx = torch.tensor([-1., 0., 1., -2., 0., 2., -1., 0., 1.]).view(3, 3)
    Gy = Gx.transpose(1, 0)
    kx = torch.stack([Gx] * channel).unsqueeze(1)
    ky = torch.stack([Gy] * channel).unsqueeze(1)
    if not nocuda:
        kx = kx.cuda()
        ky = ky.cuda()


    Ix = F.conv2d(input = imten, stride=1,weight = kx, padding = 1, groups = channel)
    Iy = F.conv2d(input = imten, stride=1,weight = ky, padding = 1, groups = channel)
    return Ix, Iy

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.)

import numpy as np
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    fig = plt.figure(figsize = (12,4))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return fig

# PURE PYTORCH
tobit = lambda x: bin(x).replace('0b','').zfill(8)
trans = lambda tmp : int(f"0b{''.join([*map(lambda x:tobit(x),tmp)])}" , 2)


P = lambda x,size: ToTensor()((ToPILImage()(x)).resize(size)).unsqueeze(0)


def ResizeImage(image, size):
    return torch.cat([*map(lambda x:P(x,size), image)], 0)

def flow2rgb(flow_map, scaled = True):
    # EXPECTED : B,C,H,W
    B, C, H, W = flow_map.size()
    condition = (flow_map.select(1, 0) == 0) & (flow_map.select(1, 1) == 0)
    condition = condition.view(B, -1).repeat(1, 2).view(*flow_map.size())
    fill = torch.ones_like(flow_map) * -1e9
    flow_map = torch.where(condition, fill, flow_map)

    if not scaled:
        flow_map = flow_map / flow_map.max()

    rgb_map = torch.ones(B, 3, H, W)
    r = flow_map.select(1, 0)
    g = -0.5 * (flow_map.select(1, 1) + r)
    b = flow_map.select(1, 1)
    rgb = torch.stack([r, g, b], 1)
    try:
        rgb_map = rgb_map + rgb
    except:
        rgb_map = rgb_map.cuda() + rgb.cuda()
    return rgb_map.clamp(0, 1)


def getflow(path):
    with open(path, mode='r') as flo:
        tag = np.fromfile(flo, np.float32, count=1)[0]
        width = np.fromfile(flo, np.int32, count=1)[0]
        height = np.fromfile(flo, np.int32, count=1)[0]
        nbands = 2
        tmp = np.fromfile(flo, np.float32, count=nbands * width * height)
        flow = np.resize(tmp, (int(height), int(width), int(nbands)))
    return flow

def warper(flow, img, scaled=True,nocuda = False):
    flow = flow.permute(0,2,3,1)
    b, h, w, c = flow.size()
    if not scaled:
        if nocuda:
            flow = flow / torch.tensor([436, 1024])
        else:
            flow = flow / torch.tensor([436, 1024]).cuda()


    meshgrid = torch.cat([torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1).expand(b, h, w, 1),
                          torch.linspace(-1.0, 1.0, h).view(1, h, 1, 1).expand(b, h, w, 1)], -1)

    if nocuda:
        grid = (meshgrid + flow)
    else:
        grid = (meshgrid.cuda() + flow)

    warped = F.grid_sample(input=img, grid=grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped

def replicatechannel(x):
    """
    :param x: [B,1,D,H,W]
    :type x: an image with channel 1
    :return: [B,3,H,W]
    :rtype: an image with replicated first channel
    """



    return x.view(1, -1).repeat(3, 1).view(3, x.size(0), x.size(2), x.size(3)).permute(1, 0, 2, 3)

def getSintelPairFrame(root, sample=None, test=False, n=2):
    def nsample(datalist):
        return random.sample(datalist, sample) if sample else datalist

    def chunker(lst):
        return [[lst[i + j] for j in range(n)] for i in range(len(lst) - (n - 1))]

    def getid(x):
        return [*map(int, re.findall(r'\d+', x.name))][0]

    def getdirdata(x):
        return [*map(lambda y: y.as_posix(), sorted(x.glob('*.png'), key=getid))]

    def getflowpath(f):
        return f.replace('final', 'flow').replace('.png', '.flo')

    def getmotionpath(f):
        f = f.replace('final', 'motion')
        root, name = f.split('frame_')
        name = str(int(name.replace('.png','')))+'.npy'
        return root + name

    def getocclusionpath(f):
        return f.replace('final', 'occlusions')

    subroot = Path(root).glob('*')

    datalist = chain.from_iterable(map(chunker, map(getdirdata, subroot)))
    if test: return nsample([{'frame': frames} for frames in datalist])
    datalist = [
        {'motion':[*map(getmotionpath, frames[:-1])], 'flow': [*map(getflowpath, frames[:-1])], 'occlusion': [*map(getocclusionpath, frames[:-1])], 'frame': frames}
        for frames in datalist]
    return nsample(datalist)


def get_train_val_test_list(trainpath="/data/keshav/sintel/training/final",
                            testpath="/data/keshav/sintel/test/final/",
                            n=10,
                            percent=0.05,
                            random_seed=0):
    from random import shuffle, seed
    from math import ceil

    seed(random_seed)

    trainpath = getSintelPairFrame(trainpath, n=n)
    testset = getSintelPairFrame(testpath, test=True, n=n)

    data = {}

    for i in trainpath:
        key = i['flow'][0].split('/flow/')[-1].split('/frame')[0]
        if key not in data:
            data[key] = [i]
        else:
            data[key].append(i)
    data = list(data.values())

    trainset = []
    valset = []
    for d in data:
        testn = ceil(len(d) * percent)  # Only 5%
        shuffle(d)
        trainset.extend(d[testn:])
        valset.extend(d[:testn])

    return trainset, valset, testset

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



import torch

def length_sq(x):
    return torch.pow(x, 2).sum(1).unsqueeze(1)

def computeocclusion(flow_fw, flow_bw):

    flow_bw_warped = warper(flow_fw, flow_bw)
    flow_fw_warped = warper(flow_bw, flow_fw)

    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)
    mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)
    occ_thresh_fw =  0.01 * mag_sq_fw + 0.5
    occ_thresh_bw =  0.01 * mag_sq_bw + 0.5

    occ_fw = (length_sq(flow_diff_fw) > occ_thresh_fw).float()
    occ_bw = (length_sq(flow_diff_bw) > occ_thresh_bw).float()

    return occ_fw, occ_bw

