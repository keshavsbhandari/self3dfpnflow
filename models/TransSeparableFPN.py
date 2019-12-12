# PYRAMID separable FOR SINTEL

import torch.nn as nn
import torch.nn.functional as F
import torch

#
#     def __init__(self, in_channels=3, out_channels=32, dilation=1, kernel_size=3, padding=1, stride=1, bias=True):
#         super(DConv2d, self).__init__()
#         kernels_per_layer = in_channels // out_channels if in_channels > out_channels else out_channels // in_channels
#         self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size,
#                                    padding=padding, groups=2, dilation=dilation, stride=stride, bias=bias)
#         self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, out_channels, kernel_size=1)

#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out


#
#     def __init__(self, in_channels=16, out_channels=32, dilation=1, kernel_size=3, padding=1, stride=1, bias=True,
#                  output_padding=(0, 0)):
#         super(DConvTranspose2d, self).__init__()
#         kernels_per_layer = in_channels // out_channels if in_channels > out_channels else out_channels // in_channels
#         self.depthwise = nn.ConvTranspose2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size,
#                                             padding=padding, groups=2,
#                                             stride=stride, dilation=dilation, bias=bias, output_padding=output_padding)
#         self.pointwise = nn.ConvTranspose2d(in_channels * kernels_per_layer, out_channels, kernel_size=1)

#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out

from torch.nn import Conv2d as DConv2d
from torch.nn import ConvTranspose2d as DConvTranspose2d


class InputDownsampler(nn.Module):
    def __init__(self, in_channels=12, out_channels=16):
        super(InputDownsampler, self).__init__()
        self.downsample = nn.Sequential(
            DConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5), stride=1,
                    padding=(1, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(0, 1)),

        )

    def forward(self, x):
        return self.downsample(x)


class OutputUpsampler(nn.Module):
    def __init__(self, in_channels=16, out_channels=6):
        super(OutputUpsampler, self).__init__()
        mid_channels = (in_channels + out_channels) // 2
        mid_channels = mid_channels + 1 if mid_channels % 2 != 0 else mid_channels
        self.upsample = nn.Sequential(
            DConvTranspose2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(5, 3), stride=1,
                             padding=(1, 1), output_padding=(0, 0)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            # nn.Dropout(0.5),
            DConvTranspose2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(5, 5),
                             stride=(2, 2),
                             padding=2, output_padding=1),

            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels, eps=1e-1, momentum=1e-5),
            nn.ReLU(),
            # nn.ReLU(inplace = True),
            # nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.upsample(x)


class FlowPrediction(nn.Module):
    def __init__(self, in_channels=16, out_channels=2):
        super(FlowPrediction, self).__init__()
        self.predict = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels, eps=1e-01, momentum=1e-5),
            # nn.ReLU(),
            nn.ReLU(),
            # nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.predict(x)


class Downsampler(nn.Module):
    def __init__(self, in_channels=16, out_channels=32):
        super(Downsampler, self).__init__()
        mid_channels = (in_channels + out_channels) // 2
        mid_channels = mid_channels + 1 if mid_channels % 2 != 0 else mid_channels
        self.seq = nn.Sequential(
            DConv2d(kernel_size=3, in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.BatchNorm2d(mid_channels),  # added batchnorm
            nn.ReLU(),
            DConv2d(kernel_size=3, in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(out_channels),  # added batchnorm
            # nn.ReLU(inplace=True),
            nn.ReLU(),

        )

        self.pool = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)
        self.onebyone = nn.Conv2d(kernel_size=1, stride=1, in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.seq(x)
        bypass = self.onebyone(x)
        x = self.pool(x)
        return x, bypass


class Upsampler(nn.Module):
    def __init__(self, in_channels=16, out_channels=32):
        super(Upsampler, self).__init__()

        mid_channels = (in_channels + out_channels) // 2
        self.seq = nn.Sequential(
            DConvTranspose2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=(2, 2),
                             padding=1, output_padding=(1, 1)),

            nn.BatchNorm2d(mid_channels),  # added batchnorm
            nn.ReLU(),
            DConv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),  # added batchnorm
            nn.ReLU(),

        )

    def forward(self, x, bypass):
        return self.seq(self.center_crop_3d(x, bypass))

    @staticmethod
    def center_crop_3d(x, X):
        _, _, height, width = x.size()
        crop_h = torch.FloatTensor([X.size()[2]]).sub(height).div(-2)
        crop_w = torch.FloatTensor([X.size()[3]]).sub(width).div(-2)
        X = F.pad(X,
                  [
                      crop_w.ceil().int()[0], crop_w.floor().int()[0],
                      crop_h.ceil().int()[0], crop_h.floor().int()[0],
                  ]
                  )
        return x + X


class Latent(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        mid_channels = (in_channels + out_channels) // 2
        super(Latent, self).__init__()
        self.latent = nn.Sequential(
            DConv2d(kernel_size=3, in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            # nn.Dropout(0.5),
            DConv2d(kernel_size=3, in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.latent(x)


class PyramidUNet(nn.Module):
    def __init__(self, in_channel=6, out_channels=2, init_feature=16, use_cst=False):
        super(PyramidUNet, self).__init__()

        self.input = InputDownsampler(in_channel, init_feature)
        self.cstinput = InputDownsampler(2, init_feature)

        self.downsample_0 = Downsampler(init_feature, init_feature * 2)
        self.downsample_1 = Downsampler(init_feature * 2, init_feature * 4)
        self.downsample_2 = Downsampler(init_feature * 4, init_feature * 8)

        self.latent = Latent(init_feature * 8, init_feature * 8)
        self.latentflow_f = FlowPrediction(init_feature * 8, out_channels)
        self.latentflow_b = FlowPrediction(init_feature * 8, out_channels)

        self.upsample_2 = Upsampler(init_feature * 8, init_feature * 4)
        self.pyramidflow_2f = FlowPrediction(init_feature * 4, out_channels)
        self.pyramidflow_2b = FlowPrediction(init_feature * 4, out_channels)

        self.upsample_1 = Upsampler(init_feature * 4, init_feature * 2)
        self.pyramidflow_1f = FlowPrediction(init_feature * 2, out_channels)
        self.pyramidflow_1b = FlowPrediction(init_feature * 2, out_channels)

        self.upsample_0 = Upsampler(init_feature * 2, init_feature)

        self.finalflow_f = OutputUpsampler(init_feature, out_channels)
        self.finalflow_b = OutputUpsampler(init_feature, out_channels)

        # self.a = nn.Parameter(torch.tensor([-0.0001]))
        # self.b = nn.Parameter(torch.tensor([0.0001]))

        if use_cst:
            self.unfold = torch.nn.Unfold(kernel_size=3, padding=1)
            self.cst = self.cst3d
        else:
            self.cst = lambda x: x

    def intpolandcat(self, ff, out, fb):
        if out.size()[2:] != ff.size()[2:]:
            ff_ = F.interpolate(ff, size=(out.size(2), out.size(3)))
            fb_ = F.interpolate(fb, size=(out.size(2), out.size(3)))
            return torch.cat([ff_, out, fb_], 1)
        else:
            return torch.cat([ff, out, fb], 1)

    def forward(self, x,motion = None, ff=None, fb=None):
        # ff, fb are 2x108x256
        # x = self.cstd(x)
        # out = torch.cat([motion, x], 1)
        x = self.input(x)
        out = self.cstinput(motion)

        out = x * out
        # out = self.intpolandcat(motion, x)

        out_0, bypass_0 = self.downsample_0(out)
        # out_0 = self.intpolandcat(ff, out_0, fb)

        out_1, bypass_1 = self.downsample_1(out_0)
        # out_1 = self.intpolandcat(ff, out_1, fb)

        out_2, bypass_2 = self.downsample_2(out_1)
        # out_2 = self.intpolandcat(ff, out_2, fb)

        latentout = self.latent(out_2)

        pyramid_2 = self.upsample_2(latentout, bypass_2)
        pyramid_1 = self.upsample_1(pyramid_2, bypass_1)
        pyramid_0 = self.upsample_0(pyramid_1, bypass_0)

        finalflow_b = self.finalflow_b(pyramid_0)
        finalflow_f = self.finalflow_f(pyramid_0)


        threshold = 1.

        finalflow = (finalflow_f * threshold, finalflow_b * threshold)

        if self.training:
            latentflow_f = self.latentflow_f(latentout)
            latentflow_b = self.latentflow_b(latentout)

            pyramidflow_2f = self.pyramidflow_2f(pyramid_2)
            pyramidflow_2b = self.pyramidflow_2b(pyramid_2)

            pyramidflow_1f = self.pyramidflow_1f(pyramid_1)
            pyramidflow_1b = self.pyramidflow_1b(pyramid_1)

            ## scale threshold
            latentflow = (latentflow_f * threshold, latentflow_b * threshold)
            pyramidflow_2 = (pyramidflow_2f * threshold, pyramidflow_2b * threshold)
            pyramidflow_1 = (pyramidflow_1f * threshold, pyramidflow_1b * threshold)

            # flows = (finalflow, pyramidflow_1, pyramidflow_2, latentflow)

            flows = (latentflow,pyramidflow_2, pyramidflow_1, finalflow)

            return flows

        else:
            return finalflow

    def cstd(self, frames):
        B, C, H, W = frames.size()
        unfold = torch.nn.Unfold(kernel_size=3, stride=1, padding=1)
        fold = torch.nn.Fold(output_size=(H, W), kernel_size=1, stride=1)
        img = unfold(frames).view(B, C, 9, -1).permute(0, 1, 3, 2)
        mid = img[:, :, :, 4].unsqueeze(-1).cuda()
        encoding = (img >= mid).float().view(-1, 9) * torch.tensor([128., 64., 32., 16., 0., 8., 4., 2., 1.]).cuda()
        encoding = encoding.sum(-1).view(B, C, -1) / 255.
        return fold(encoding)

    # def shrinker(self, x):
    #     """
    #                 (b-a)(x - min)
    #         f(x) = ------------------  + a
    #                 max - min
    #     :param x:
    #     :type x:
    #     :return:
    #     :rtype:
    #     """
    #     return self.a + (((self.b - self.a)*(x - x.min()))/ (x.max() - x.min() + 1))
