#TORCH IMPORTS
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import functional as F
import torch

#EXTRA LIBS
from tqdm import tqdm

#LOCAL IMPORTS
from dataloader.sintelloader3d import SintelLoader3D
from models.Pyramid3dnet import PyramidUNet
from utils import (warper, AverageMeter, flow2rgb, replicatechannel,computeocclusion, ResizeImage)
from torchvision.transforms import (ToTensor, ToPILImage)

class FlowTrainer(object):
    def __init__(self):
        super(FlowTrainer, self).__init__()
        # not the best model...
        self.model = PyramidUNet()

        self.epoch = 1000
        self.dataloader = SintelLoader3D()
        self.gpu_ids = [0, 1,2]

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, len(self.dataloader.train()))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.writer = SummaryWriter()
        self.global_step = 0
        self.tripletloss = torch.nn.TripletMarginLoss()
        self.load_model_path = False

    def initialize(self):
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
        if self.load_model_path:
            # LOAD MODEL WEIGHTS HERE
            pass
        self.initialized = True

    def savemodel(self, metrics, compare='val_loss'):
        # Save model in save_model_path
        if self.best_metrics.get('val_loss') > metrics.get('val_loss'):
            # save only if new metrics are low
            self.best_metrics.update(metrics)
            pass
        else:
            # Load from the best saved
            pass

    def warpframes(self, ff, fb, frame):
        ff_ = self.warper(ff, frame, 'ff')
        fb_ = self.warper(fb, frame, 'fb')
        warpframe = (ff_, fb_)
        occlusion = self.occwarper(ff, fb)
        return occlusion, warpframe

    #     def warpocclusion(self, ff, fb):
    #         return self.occwarper(ff,fb)

    def train(self, nb_epoch):
        trainstream = tqdm(self.dataloader.train())
        self.avg_loss = AverageMeter()
        self.avg_epe = AverageMeter()
        self.model.train()
        for i, data in enumerate(trainstream):
            self.global_step += 1
            trainstream.set_description('TRAINING')

            # GET X and Frame 2
            # wdt = data['displacement'].to(self.device)
            frame = data['frame'].to(self.device)
            flow = data['flow'].cpu()
            # frame.requires_grad = True
            flow.requires_grad = False
            """
            NOTE : THIS MUST BE ADJUSTED AT DATA LOADER SIDE 
            torch.Size([1, 2, 9, 436, 1024])    -> finalflow size
            torch.Size([1, 2, 9, 108, 256])     -> pyraflow1 size
            torch.Size([1, 2, 9, 54, 128])      -> pyraflow2 size
            torch.Size([1, 2, 9, 27, 64])       -> pyraflow3 size
            """
            pyra1_frame = data['pyra1_frame'].to(self.device)
            # pyra1_frame.requires_grad = True
            pyra2_frame = data['pyra2_frame'].to(self.device)
            # pyra2_frame.requires_grad = True
            laten_frame = data['laten_frame'].to(self.device)
            # laten_frame.requires_grad = True

            self.optimizer.zero_grad()
            # forward
            with torch.set_grad_enabled(True):
                finalflow, pyraflow1, pyraflow2, latenflow = self.model(frame)
                occlu_final, frame_final = self.warpframes(*finalflow, frame)
                occlu_pyra1, frame_pyra1 = self.warpframes(*pyraflow1, pyra1_frame)
                occlu_pyra2, frame_pyra2 = self.warpframes(*pyraflow2, pyra2_frame)
                occlu_laten, frame_laten = self.warpframes(*latenflow, laten_frame)

                # print(occlu_final[0].shape)

                cost_final = self.getcost(*frame_final, *occlu_final, frame)
                cost_pyra1 = self.getcost(*frame_pyra1, *occlu_pyra1, pyra1_frame)
                cost_pyra2 = self.getcost(*frame_pyra2, *occlu_pyra2, pyra2_frame)
                cost_laten = self.getcost(*frame_laten, *occlu_laten, laten_frame)

                eper_final = self.epe(finalflow[1].cpu().detach(), flow.cpu().detach())

                loss = cost_final + cost_pyra1 + cost_pyra2 + cost_laten

                self.avg_loss.update(loss.item(), i + 1)
                self.avg_epe.update(eper_final.item(), i + 1)

                loss.backward()

                self.optimizer.step()

                self.writer.add_scalar('Loss/train',
                                       self.avg_loss.avg, self.global_step)

                self.writer.add_scalar('EPE/train',
                                       self.avg_epe.avg, self.global_step)

                trainstream.set_postfix({'epoch': nb_epoch,
                                         'loss': self.avg_loss.avg,
                                         'epe': self.avg_epe.avg})
        self.scheduler.step(loss)
        trainstream.close()

        fb_frame_final = frame_final[1]
        fb_final = finalflow[1]
        fb_occlu_final = occlu_final[1]

        return self.train_epoch_end({'TRloss': self.avg_loss.avg,
                                     'epoch': nb_epoch,
                                     'pred_frame': fb_frame_final[0, :, 0:4, :].permute(1, 0, 2, 3),
                                     'gt_frame': frame[0, :, 0:4, :].permute(1, 0, 2, 3),
                                     'pred_flow': flow2rgb(fb_final[0, :, 0:4, :].permute(1, 0, 2, 3) * torch.tensor([436./260., 1024./256.]).view(1,2,1,1).cuda()),
                                     'gt_flow': flow2rgb(flow[0, :, 0:4, :].permute(1, 0, 2, 3)),
                                     'pred_occ': fb_occlu_final[0, :, 0:4, :].permute(1, 0, 2, 3),
                                     'gt_occ': data['occlusion'][0, :, 0:4, :].permute(1, 0, 2, 3)})

    def train_epoch_end(self, metrics):
        self.model.eval()
        with torch.no_grad():
            pred_frame = metrics.get('pred_frame')
            gt_frame = metrics.get('gt_frame')
            pred_flow = metrics.get('pred_flow')
            gt_flow = metrics.get('gt_flow')
            pred_occ = replicatechannel(metrics.get('pred_occ'))
            gt_occ = replicatechannel(metrics.get('gt_occ'))

            data = torch.stack([pred_frame.cuda(), gt_frame.cuda(), pred_flow.cuda(), gt_flow.cuda(), pred_occ.cuda(), gt_occ.cuda()], 0)

            data = data.reshape(-1,3,260,256).cpu()

            grid = ToTensor()((ToPILImage()(make_grid(data, nrow=4))))
            self.writer.add_images('TRAIN/Results', grid.unsqueeze(0), metrics.get('n_batch'))
        self.val(metrics.get('epoch'))

    def val(self, nb_epoch):
        self.model.eval()
        # if self.val_loader is None: return self.test()
        # DO VAL STUFF HERE
        valstream = tqdm(self.dataloader.val())
        self.avg_loss = AverageMeter()
        self.avg_epe = AverageMeter()
        valstream.set_description('VALIDATING')
        with torch.no_grad():
            for i, data in enumerate(valstream):
                frame = data['frame'].to(self.device)
                flow = data['flow'].cpu()
                finalflow = self.model(frame)
                occlu_final, frame_final = self.warpframes(*finalflow, frame)
                loss = self.getcost(*frame_final, *occlu_final, frame)
                eper_final = self.epe(flow.cpu().detach(), finalflow[1].cpu().detach())
                self.avg_loss.update(loss.item(), i + 1)
                self.avg_epe.update(eper_final.item(), i + 1)

        self.writer.add_scalar('Loss/val',
                               self.avg_loss.avg, self.global_step)

        self.writer.add_scalar('EPE/val',
                               self.avg_epe.avg, self.global_step)

        fb_frame_final = frame_final[1]
        fb_final = finalflow[1]
        fb_occlu_final = occlu_final[1]

        valstream.close()

        self.val_end({'VLloss': self.avg_loss.avg,
                      'epoch': nb_epoch,
                      'pred_frame': fb_frame_final[0, :, 0:4, :].permute(1, 0, 2, 3),
                      'gt_frame': frame[0, :, 0:4, :].permute(1, 0, 2, 3),
                      'pred_flow': flow2rgb(fb_final[0, :, 0:4, :].permute(1, 0, 2, 3) * torch.tensor([436./260., 1024./256.]).view(1,2,1,1).cuda()),
                      'gt_flow': flow2rgb(flow[0, :, 0:4, :].permute(1, 0, 2, 3)),
                      'pred_occ': fb_occlu_final[0, :, 0:4, :].permute(1, 0, 2, 3),
                      'gt_occ': data['occlusion'][0, :, 0:4, :].permute(1, 0, 2, 3)})

    def val_end(self, metrics):
        self.model.eval()
        with torch.no_grad():
            pred_frame = metrics.get('pred_frame').cpu()
            gt_frame = metrics.get('gt_frame').cpu()
            pred_flow = metrics.get('pred_flow').cpu()
            gt_flow = metrics.get('gt_flow').cpu()
            pred_occ = replicatechannel(metrics.get('pred_occ')).cpu()
            gt_occ = replicatechannel(metrics.get('gt_occ')).cpu()
            data = torch.stack([pred_frame, gt_frame, pred_flow, gt_flow, pred_occ, gt_occ], 0).cpu()
            data = data.reshape(-1, 3, data.size(3), data.size(4)).cpu()
            grid = make_grid(data, nrow=4)
            self.writer.add_images('VAL/Results', grid.unsqueeze(0), metrics.get('n_batch'))
        self.test(metrics.get('epoch'))

    def test(self, nb_epoch):
        self.model.eval()
        teststream = tqdm(self.dataloader.test())
        self.avg_loss = AverageMeter()
        teststream.set_description('TESTING')
        with torch.no_grad():
            for i, data in enumerate(teststream):
                frame = data['frame']
                finalflow = self.model(frame)

                occlu_final, frame_final = self.warpframes(*finalflow, frame)
                loss = self.getcost(*frame_final, *occlu_final, frame)

                self.avg_loss.update(loss.item(), i + 1)

        self.writer.add_scalar('Loss/test',
                               self.avg_loss.avg, self.global_step)

        fb_frame_final = frame_final[1]
        fb_final = finalflow[1]
        fb_occlu_final = occlu_final[1]

        teststream.close()

        self.test_end({'VLloss': self.avg_loss.avg,
                       'epoch': nb_epoch,
                       'pred_frame': fb_frame_final[0, :, 0:4, :].permute(1, 0, 2, 3),
                       'gt_frame': frame[0, :, 0:4, :].permute(1, 0, 2, 3),
                       'pred_flow': flow2rgb(fb_final[0, :, 0:4, :].permute(1, 0, 2, 3)  * torch.tensor([436./260., 1024./256.]).view(1,2,1,1).cuda()),
                       'pred_occ': fb_occlu_final[0, :, 0:4, :].permute(1, 0, 2, 3), })

    def test_end(self, metrics):
        self.model.eval()
        with torch.no_grad():
            pred_frame = metrics.get('pred_frame').cpu()
            gt_frame = metrics.get('gt_frame').cpu()
            pred_flow = metrics.get('pred_flow').cpu()
            pred_occ = replicatechannel(metrics.get('pred_occ')).cpu()
            data = torch.stack([pred_frame, gt_frame, pred_flow, pred_occ], 0)
            data = data.reshape(-1, 3, data.size(3), data.size(4)).cpu()
            grid = make_grid(data, nrow=4)
            self.writer.add_images('Test/Results', grid.unsqueeze(0), metrics.get('n_batch'))

    def loggings(self, **metrics):
        pass

    def warper(self, flows, frames, mode='ff', scaled=True, nocuda=False):
        B, _, D, H, W = flows.size()
        dflow = flows.permute(0, 2, 1, 3, 4).reshape(-1, 2, H, W)

        if mode == 'ff':
            dframe = frames[:, :, :-1, :]  # given frame from 0 to n-1 predict frame 1 to n
        elif mode == 'fb':
            dframe = frames[:, :, 1:, :]  # given frame from 1 to n predict frame 0 to n-1
        else:
            raise Exception("Mode must be flow-forwad 'ff' or flow-backward 'fb'")

        dframe = dframe.permute(0, 2, 1, 3, 4).reshape(-1, 3, H, W)

        from termcolor import colored
        warped = warper(dflow.cuda(), dframe.cuda(), scaled=scaled, nocuda=nocuda).view(B, D, 3, H, W).permute(0, 2, 1, 3, 4).cuda()
        return warped

    def occwarper(self, ff, fb):
        B, C, D, H, W = ff.size()
        dff = ff.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        dfb = fb.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

        ff_occ, fb_occ = computeocclusion(dff, dfb)

        ff_occ = ff_occ.view(B, D, 1, H, W).permute(0, 2, 1, 3, 4)
        fb_occ = fb_occ.view(B, D, 1, H, W).permute(0, 2, 1, 3, 4)

        # print(ff_occ.shape)

        return ff_occ, fb_occ

    def log_triplet_loss(self, anchor, positive, negative, mask, q=1e-4):
        pos = torch.mul(torch.pow((torch.abs(anchor - positive) + 1e-2), q), mask)
        neg = torch.mul(torch.pow((torch.abs(anchor - positive) + 1e-2), q), mask)
        loss = torch.log(torch.exp(pos / (neg + 1e-10)))
        loss = loss.sum() / (mask.sum() + 1e-10)
        return loss

    def getcost(self, ff_frame, fb_frame, ff_occlu, fb_occlu, frame):

        ff_frame = ff_frame.cuda()
        fb_frame = fb_frame.cuda()
        frame = frame.cuda()

        ff_truth, fb_truth = frame[:, :, 1:, :], frame[:, :, :-1, :]

        ff_ploss = self.log_triplet_loss(ff_frame, ff_truth, fb_truth, 1. - ff_occlu)
        fb_ploss = self.log_triplet_loss(fb_frame, fb_truth, ff_truth, 1. - fb_occlu)

        ff_tloss = self.tripletloss(ff_frame, ff_truth, fb_truth)
        fb_tloss = self.tripletloss(fb_frame, fb_truth, ff_truth)

        total = ff_ploss + fb_ploss + ff_tloss + fb_tloss

        # total = ff_tloss + fb_tloss

        return total

    def epe(self, source, target):
        with torch.no_grad():
            source = source.cpu().detach() / torch.tensor([260.,256.]).view(1,2,1,1,1)
            target = target.cpu().detach() / torch.tensor([436.,1024.]).view(1,2,1,1,1)
            # from termcolor import colored
            # print(colored(f'{source.shape, target.shape, source.max(), target.max()}','red'))
            B, C, D, H, W = source.size()
            diff = (source - target).permute(0, 2, 1, 3, 4).reshape(-1, C * H * W)
            return torch.norm(diff, p=2, dim=1).mean()

    def run(self):
        self.initialize()
        for i in range(self.epoch):
            self.train(i)
        self.writer.close()