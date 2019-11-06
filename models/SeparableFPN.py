# PYRAMID separable FOR SINTEL

import torch.nn as nn
import torch.nn.functional as F
import torch

import torch.nn as nn


class DConv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, dilation=1, kernel_size=3, padding=1, stride=1, bias=True):
        super(DConv2d, self).__init__()
        kernels_per_layer = in_channels // out_channels if in_channels > out_channels else out_channels // in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size,
                                   padding=padding, groups=2, dilation=dilation, stride=stride, bias=bias)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DConvTranspose2d(nn.Module):
    def __init__(self, in_channels=16, out_channels=32, dilation=1, kernel_size=3, padding=1, stride=1, bias=True,
                 output_padding=(0, 0)):
        super(DConvTranspose2d, self).__init__()
        kernels_per_layer = in_channels // out_channels if in_channels > out_channels else out_channels // in_channels
        self.depthwise = nn.ConvTranspose2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size,
                                            padding=padding, groups=2,
                                            stride=stride, dilation=dilation, bias=bias, output_padding=output_padding)
        self.pointwise = nn.ConvTranspose2d(in_channels * kernels_per_layer, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class InputDownsampler(nn.Module):
    def __init__(self, in_channels=12, out_channels=16, depth=10):
        super(InputDownsampler, self).__init__()
        self.downsample = nn.Sequential(
            DConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5), stride=1,
                    padding=(1, 2)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(0, 1)),
        )

    def forward(self, x):
        return self.downsample(x)


class OutputUpsampler(nn.Module):
    def __init__(self, in_channels=16, out_channels=6, depth=10):
        super(OutputUpsampler, self).__init__()
        mid_channels = (in_channels + out_channels) // 2
        mid_channels = mid_channels + 1 if mid_channels % 2 != 0 else mid_channels
        self.upsample = nn.Sequential(
            DConvTranspose2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(5, 3), stride=1,
                             padding=(1, 1), output_padding=(0, 0)),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(mid_channels),
            DConvTranspose2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(5, 5),
                             stride=(2, 2),
                             padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.upsample(x)


class FlowPrediction(nn.Module):
    def __init__(self, in_channels=16, out_channels=2, depth=10):
        super(FlowPrediction, self).__init__()
        self.predict = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(), )

    def forward(self, x):
        return self.predict(x)


class Downsampler(nn.Module):
    def __init__(self, in_channels=16, out_channels=32, depth=10):
        super(Downsampler, self).__init__()
        mid_channels = (in_channels + out_channels) // 2
        mid_channels = mid_channel + 1 if mid_channels % 2 != 0 else mid_channels
        self.seq = nn.Sequential(
            DConv2d(kernel_size=3, in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.LeakyReLU(),
            DConv2d(kernel_size=3, in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.LeakyReLU(),
        )

        self.pool = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)
        self.onebyone = nn.Conv2d(kernel_size=1, stride=1, in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.seq(x)
        bypass = self.onebyone(x)
        x = self.pool(x)
        return x, bypass


class Upsampler(nn.Module):
    def __init__(self, in_channels=16, out_channels=32, depth=10):
        super(Upsampler, self).__init__()

        mid_channels = (in_channels + out_channels) // 2

        self.seq = nn.Sequential(
            DConvTranspose2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=(2, 2),
                             padding=1, output_padding=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(mid_channels),
            DConv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
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
    def __init__(self, in_channels=256, out_channels=256, depth=10):
        mid_channels = (in_channels + out_channels) // 2
        super(Latent, self).__init__()
        self.latent = nn.Sequential(
            DConv2d(kernel_size=3, in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(mid_channels),
            DConv2d(kernel_size=3, in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.latent(x)


class PyramidUNet(nn.Module):
    def __init__(self, in_channel=12, out_channels=2, init_feature=16, depth=10, use_cst=False):
        super(PyramidUNet, self).__init__()

        self.input = InputDownsampler(in_channel, init_feature)
        self.cstinput = InputDownsampler(in_channel, init_feature, depth)

        self.downsample_0 = Downsampler(init_feature, init_feature * 2)
        self.downsample_1 = Downsampler(init_feature * 2, init_feature * 4)
        self.downsample_2 = Downsampler(init_feature * 4, init_feature * 8)

        self.latent = Latent(init_feature * 8, init_feature * 8, depth)
        self.latentflow_f = FlowPrediction(init_feature * 8, out_channels)
        self.latentflow_b = FlowPrediction(init_feature * 8, out_channels)

        self.upsample_2 = Upsampler(init_feature * 8, init_feature * 4)
        self.pyramidflow_2f = FlowPrediction(init_feature * 4, out_channels)
        self.pyramidflow_2b = FlowPrediction(init_feature * 4, out_channels)

        self.upsample_1 = Upsampler(init_feature * 4, init_feature * 2)
        self.pyramidflow_1f = FlowPrediction(init_feature * 2, out_channels)
        self.pyramidflow_1b = FlowPrediction(init_feature * 2, out_channels)

        self.upsample_0 = Upsampler(init_feature * 2, init_feature)
        self.pyramidflow_0f = FlowPrediction(init_feature, out_channels)
        self.pyramidflow_0b = FlowPrediction(init_feature, out_channels)

        self.finalflow_f = OutputUpsampler(init_feature, out_channels)
        self.finalflow_b = OutputUpsampler(init_feature, out_channels)

        if use_cst:
            self.unfold = torch.nn.Unfold(kernel_size=3, padding=1)
            self.cst = self.cst3d
        else:
            self.cst = lambda x: x

    def forward(self, x):
        """
        :param x: input image [B X C X 10 X 436 X 1024] : FOR SINTEL CONFIG
        :type x:
        :return: 4 torch tensor tuples, if training else forwardfinal flow
        :rtype: [torch.Size([1, 2, 9, 436, 1024]) -> forward_final_flow
                torch.Size([1, 2, 9, 436, 1024]) -> backward_final_flow]

                [torch.Size([1, 2, 9, 108, 256]) -> forward_pyramid_1_flow
                torch.Size([1, 2, 9, 108, 256]) -> backward_pyramid_1_flow]

                [torch.Size([1, 2, 9, 54, 128]) -> forward_pyramid_2_flow
                torch.Size([1, 2, 9, 54, 128])-> backward_pyramid_2_flow]

                [torch.Size([1, 2, 9, 27, 64]) -> forward_latent_flow
                torch.Size([1, 2, 9, 27, 64]) -> backward_latent_flow]
        """


        cstx = self.cst(x)

        stackx = torch.cat([x, x[:, 3:6], x[:, 0:3]], 1)
        stckct = torch.cat([cstx, cstx[:, 3:6], cstx[:, 0:3]], 1)

        outin = self.input(stackx)
        outcst = self.cstinput(stckct)

        out = outin + outcst

        # out = self.cstinput(x)
        out_0, bypass_0 = self.downsample_0(out)
        out_1, bypass_1 = self.downsample_1(out_0)
        out_2, bypass_2 = self.downsample_2(out_1)

        latentout = self.latent(out_2)

        pyramid_2 = self.upsample_2(latentout, bypass_2)
        pyramid_1 = self.upsample_1(pyramid_2, bypass_1)
        pyramid_0 = self.upsample_0(pyramid_1, bypass_0)

        finalflow_b = self.finalflow_f(pyramid_0)

        finalflow_f = self.finalflow_f(pyramid_0)

        finalflow = (finalflow_f, finalflow_b)

        if self.training:
            latentflow_f = self.latentflow_f(latentout)
            # latentflow_f = latentflow_f * scaler

            latentflow_b = self.latentflow_b(latentout)
            # latentflow_b = latentflow_b * scaler

            pyramidflow_1f = self.pyramidflow_1f(pyramid_1)
            # pyramidflow_1f = pyramidflow_1f * scaler
            pyramidflow_1b = self.pyramidflow_1b(pyramid_1)
            # pyramidflow_1b = pyramidflow_1b * scaler

            pyramidflow_2f = self.pyramidflow_2f(pyramid_2)
            # pyramidflow_2f = pyramidflow_2f * scaler
            pyramidflow_2b = self.pyramidflow_2b(pyramid_2)
            # pyramidflow_2b = pyramidflow_2b * scaler

            latentflow = (latentflow_f, latentflow_b)
            pyramidflow_2 = (pyramidflow_2f, pyramidflow_2b)
            pyramidflow_1 = (pyramidflow_1f, pyramidflow_1b)

            flows = (finalflow, pyramidflow_1, pyramidflow_2, latentflow)

            return flows

        else:
            return finalflow

    def cstd(self, frames):
        B, C, H, W = frames.size()
        unfold = torch.nn.Unfold(kernel_size=3, stride=1, padding=1)
        fold = torch.nn.Fold(output_size=(H, W), kernel_size=1, stride=1)
        img = unfold(frames).view(B, C, 9, -1).permute(0, 1, 3, 2)
        mid = img[:, :, :, 4].unsqueeze(-1)
        encoding = (img >= mid).float().view(-1, 9) * torch.tensor([128., 64., 32., 16., 0., 8., 4., 2., 1.])
        encoding = encoding.sum(-1).view(B, C, -1) / 255.
        return fold(encoding)