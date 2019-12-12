#TORCH IMPORTS
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import functional as F
import torch
import os
#EXTRA LIBS
from tqdm import tqdm

#LOCAL IMPORTS
from dataloader.transloader import SintelLoader
from models.TransSeparableFPN import PyramidUNet
from utils import (warper, AverageMeter, flow2rgb, replicatechannel,computeocclusion, ResizeImage, plot_grad_flow, weights_init, getImGradient)
from torchvision.transforms import (ToTensor, ToPILImage)


from os import environ as env
GPUS = env["CUDA_VISIBLE_DEVICES"]
GPUS_LIST = list(range(len(GPUS.split(','))))

print(GPUS_LIST)
#
class FlowTrainer(object):
    def __init__(self):
        super(FlowTrainer, self).__init__()
        # not the best model...
        self.model = PyramidUNet()

        self.epoch = 1000
        self.dataloader = SintelLoader()
        self.gpu_ids = GPUS_LIST

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, len(self.dataloader.train()))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.writer = SummaryWriter()
        self.global_step = 0
        self.tripletloss = torch.nn.TripletMarginLoss()
        self.load_model_path = "./archive/best.pth"
        self.stat_cache = None

    def initialize(self):
        self.model.apply(weights_init)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
        if self.load_model_path:
            # LOAD MODEL WEIGHTS HERE
            if os.path.exists(self.load_model_path):
                self.load_old_best()
        self.initialized = True

    def savemodel(self, metrics):
        import json
        with open('./archive/metrics.txt','w') as f:
            json.dump(metrics,f)
        torch.save(self.model.module.state_dict(), self.load_model_path)

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

            pyra1_frame = F.interpolate(frame, size=(108, 256))
            pyra2_frame = F.interpolate(frame, size=(54, 128))
            laten_frame = F.interpolate(frame, size=(27, 64))

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
            # pyra1_frame = data['pyra1_frame'].to(self.device)
            # pyra1_frame.requires_grad = True
            # pyra2_frame = data['pyra2_frame'].to(self.device)
            # pyra2_frame.requires_grad = True
            # laten_frame = data['laten_frame'].to(self.device)
            # laten_frame.requires_grad = True
            # ff = data['ff'].to(self.device)
            # fb = data['fb'].to(self.device)
            motion = data['motion'].to(self.device)
            subf = data['subf'].to(self.device)
            # forward
            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()
                # flows = self.model(frame, ff, fb)
                # for i, finalflow in enumerate(flows):
                #     flowdim = finalflow[0].size()[2:]
                #     framedim = frame.size()[2:]
                #
                #     if flowdim != framedim:
                #         frame_ = F.interpolate(frame, size=(finalflow[0].size(2),finalflow[0].size(3)))
                #         occlu_final, frame_final = self.warpframes(*finalflow, frame_)
                #         loss = self.getcost(*frame_final, *occlu_final, frame_)
                #     else:
                #         occlu_final, frame_final = self.warpframes(*finalflow, frame)
                #         loss = self.getcost(*frame_final, *occlu_final, frame)
                #
                #     self.optimizer.zero_grad()
                #     if i < 3:
                #         loss.backward(retain_graph=True)
                #     else:
                #         loss.backward()
                #         self.writer.add_figure('Activations', plot_grad_flow(self.model.named_parameters()),
                #                                self.global_step)
                #     self.optimizer.step()
                #
                # self.avg_loss.update(loss.item(), i + 1)
                # eper_final = self.epe(finalflow[1].detach(), flow.detach())
                # self.avg_epe.update(eper_final.item(), i + 1)
                #
                # self.writer.add_scalar('Loss/train',
                #                        self.avg_loss.avg, self.global_step)
                #
                # self.writer.add_scalar('EPE/train',
                #                        self.avg_epe.avg, self.global_step)
                #
                # trainstream.set_postfix({'epoch': nb_epoch,
                #                          'loss': self.avg_loss.avg,
                #                          'epe': self.avg_epe.avg})


                # DIfferent way above

                # print(motion.shape, frame.shape)

                latenflow, pyraflow2, pyraflow1, finalflow = self.model(subf, motion)

                occlu_final, frame_final = self.warpframes(*finalflow, frame)
                occlu_pyra1, frame_pyra1 = self.warpframes(*pyraflow1, pyra1_frame)
                occlu_pyra2, frame_pyra2 = self.warpframes(*pyraflow2, pyra2_frame)
                occlu_laten, frame_laten = self.warpframes(*latenflow, laten_frame)

                # print(occlu_final[0].shape)

                cost_final = self.getcost(*frame_final, *occlu_final, frame, *finalflow)
                cost_pyra1 = self.getcost(*frame_pyra1, *occlu_pyra1, pyra1_frame, *pyraflow1)
                cost_pyra2 = self.getcost(*frame_pyra2, *occlu_pyra2, pyra2_frame, *pyraflow2)
                cost_laten = self.getcost(*frame_laten, *occlu_laten, laten_frame, *latenflow)

                loss = cost_final + cost_pyra1 + cost_pyra2 + cost_laten


                # self.optimizer.zero_grad()
                loss.backward()
                self.avg_loss.update(loss.item(), i + 1)

                self.writer.add_figure('Activations', plot_grad_flow(self.model.named_parameters()), self.global_step)

                self.optimizer.step()

                eper_final = self.epe(finalflow[1].detach(), flow.detach())
                self.avg_epe.update(eper_final.item(), i + 1)

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

        self.writer.add_histogram('REAL/flow_u', flow[0,0].view(-1), nb_epoch)
        self.writer.add_histogram('REAL/flow_v', flow[0,1].view(-1), nb_epoch)

        self.writer.add_histogram('PRED/flow_u_ff', finalflow[0][0,0].view(-1), nb_epoch)
        self.writer.add_histogram('PRED/flow_v_ff', finalflow[0][0,1].view(-1), nb_epoch)

        self.writer.add_histogram('PRED/flow_u_fb', finalflow[1][0,0].view(-1), nb_epoch)
        self.writer.add_histogram('PRED/flow_v_fb', finalflow[1][0,1].view(-1), nb_epoch)

        self.writer.add_histogram('REAL/occ',data['occlusion'][0].view(-1),nb_epoch)

        self.writer.add_histogram('PRED/occ_ff',occlu_final[0][0].view(-1),nb_epoch)
        self.writer.add_histogram('PRED/occ_fb',occlu_final[1][0].view(-1),nb_epoch)

        return self.train_epoch_end({'TRloss': self.avg_loss.avg,
                                     'epoch': nb_epoch,
                                     'pred_frame': fb_frame_final[0:4],
                                     'gt_frame': frame[0:4,:3],
                                     'subf': subf[0:4, :3],
                                     'pred_flow': flow2rgb(fb_final[0:4], False),
                                     'gt_flow': flow2rgb(flow[0:4],False),
                                     # 'ff_in': flow2rgb(ff[0:4], False),
                                     # 'fb_in': flow2rgb(fb[0:4], False),
                                     'pred_occ': 1. - fb_occlu_final[0:4],
                                     'gt_occ': data['occlusion'][0:4]})

    def train_epoch_end(self, metrics):
        self.model.eval()
        with torch.no_grad():
            pred_frame = metrics.get('pred_frame')
            gt_frame = metrics.get('gt_frame')
            subf= metrics.get('subf')
            pred_flow = metrics.get('pred_flow')
            gt_flow = metrics.get('gt_flow')

            # ff_in = metrics.get('ff_in')
            # fb_in = metrics.get('fb_in')
            pred_occ = replicatechannel(metrics.get('pred_occ'))
            gt_occ = replicatechannel(metrics.get('gt_occ'))

            data = torch.cat([pred_frame.cuda(), gt_frame.cuda(),subf.cuda(), pred_flow.cuda(), gt_flow.cuda(), pred_occ.cuda(), gt_occ.cuda()],0)
            # ff_data = make_grid(torch.cat([ff_in.cuda(),fb_in.cuda()],0).cpu(), nrow=4)
            data = data.cpu()

            grid = make_grid(data, nrow=4)
            grid = ToTensor()((ToPILImage()(grid)).resize((4106//6,2630//4)))
            self.writer.add_images('TRAIN/Results', grid.unsqueeze(0), metrics.get('n_batch'))
            # self.writer.add_images('INPUT/Flows_prior', ff_data.unsqueeze(0), metrics.get('n_batch'))

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
                motion = data['motion'].to(self.device)
                # ff = data['ff'].to(self.device)
                # fb = data['fb'].to(self.device)
                finalflow = self.model(frame, motion)
                finalflow = finalflow[::-1]
                occlu_final, frame_final = self.warpframes(*finalflow, frame)


                loss = self.getcost(*frame_final, *occlu_final, frame, *finalflow)
                eper_final = self.epe(flow.detach(), finalflow[1].detach())
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
                      'VLepe':self.avg_epe.avg,
                      'epoch': nb_epoch,
                      'pred_frame': fb_frame_final[0:4],
                      'gt_frame': frame[0:4,:3],
                      'pred_flow': flow2rgb(fb_final[0:4], False),
                      'gt_flow': flow2rgb(flow[0:4],False),
                      'pred_occ': 1. - fb_occlu_final[0:4],
                      'gt_occ': data['occlusion'][0:4]})

    def val_end(self, metrics):
        """WRITE STAT FIRST"""
        if self.stat_cache is None:
            self.stat_cache = {'VLloss': metrics.get('VLloss'),
                            'VLepe': metrics.get('VLepe')}
            self.savemodel({'VLloss': metrics.get('VLloss'),
                            'VLepe': metrics.get('VLepe')})
        else:
            if self.stat_cache.get('VLloss') < metrics.get('VLloss'):
                self.stat_cache.update({'VLloss': metrics.get('VLloss'),
                                'VLepe': metrics.get('VLepe')})
                self.savemodel(self.stat_cache)
            else:
                self.load_old_best()

        self.model.eval()
        with torch.no_grad():
            pred_frame = metrics.get('pred_frame').cpu()
            gt_frame = metrics.get('gt_frame').cpu()
            pred_flow = metrics.get('pred_flow').cpu()
            gt_flow = metrics.get('gt_flow').cpu()
            pred_occ = replicatechannel(metrics.get('pred_occ')).cpu()
            gt_occ = replicatechannel(metrics.get('gt_occ')).cpu()
            data = torch.cat([pred_frame, gt_frame, pred_flow, gt_flow, pred_occ, gt_occ], 0).cpu()
            grid = make_grid(data, nrow=3)
            grid = ToTensor()((ToPILImage()(grid)).resize((4106 // 6, 2630 // 4)))
            self.writer.add_images('VAL/Results', grid.unsqueeze(0), metrics.get('n_batch'))
        # self.test(metrics.get('epoch'))

    def load_old_best(self):
        import json
        with open('./archive/metrics.txt', 'r') as f:
            self.stat_cache = json.load(f)
        self.model.module.load_state_dict(torch.load(self.load_model_path))

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
                       'pred_flow': flow2rgb(fb_final[0, :, 0:4, :].permute(1, 0, 2, 3),False),
                       'pred_occ': 1. - fb_occlu_final[0, :, 0:4, :].permute(1, 0, 2, 3), })

    def test_end(self, metrics):
        self.model.eval()
        with torch.no_grad():
            pred_frame = metrics.get('pred_frame').cpu()
            gt_frame = metrics.get('gt_frame').cpu()
            pred_flow = metrics.get('pred_flow').cpu()
            pred_occ = replicatechannel(metrics.get('pred_occ')).cpu()
            data = torch.stack([pred_frame, gt_frame, pred_flow, pred_occ], 0).cpu()
            # data = data.reshape(-1, 3, data.size(3), data.size(4)).cpu()
            grid = make_grid(data, nrow=3)
            self.writer.add_images('Test/Results', grid.unsqueeze(0), metrics.get('n_batch'))

    def loggings(self, **metrics):
        pass

    def warper(self, flows, frames, mode='ff', scaled=True, nocuda=False):
        if mode == 'ff':
            dframe = frames[:,:3]  # given frame from 0 to n-1 predict frame 1 to n
        elif mode == 'fb':
            dframe = frames[:, 3:]  # given frame from 1 to n predict frame 0 to n-1
        else:
            raise Exception("Mode must be flow-forwad 'ff' or flow-backward 'fb'")
        warped = warper(flows.cuda(), dframe.cuda(), scaled=True, nocuda=nocuda).cuda()
        return warped

    def occwarper(self, ff, fb):
        ff_occ, fb_occ = computeocclusion(ff, fb)
        return ff_occ, fb_occ

    def log_triplet_loss(self, anchor, positive, negative, maskp, maskn, q=1e-4):
        pos = torch.mul(torch.pow((torch.abs(anchor - positive) + 1e-2), q), maskp)
        neg = torch.mul(torch.pow((torch.abs(anchor - negative) + 1e-2), q), maskn)

        pos = pos.sum() / (maskp.sum() + 1e-10)
        neg = neg.sum() / (maskn.sum()+1e-10)

        # loss = torch.log(torch.exp(pos / (neg + 1e-10)))
        # loss = loss.sum() / (mask.sum() + 1e-10)
        return pos, neg

    def getcost(self, ff_frame, fb_frame, ff_occlu, fb_occlu, frame, ff_flow, fb_flow):

        ff_frame = ff_frame.cuda()
        fb_frame = fb_frame.cuda()
        frame = frame.cuda()

        # klloss = F.kl_div(ff_frame, fb_frame)
        ff_framex, ff_framey = getImGradient(frame[:, :3])
        fb_framex, fb_framey = getImGradient(frame[:, 3:])
        # print('----BEG---')
        # print(frame[:,:3].shape, ff_flow[:,:1].shape, ff_framex.shape, ff_flow[:,1:].shape, ff_framey.shape, ff_frame.shape)
        # print('--------')
        # print(frame[:, 3:].shape, fb_flow[:, :1].shape, fb_framex.shape, fb_flow[:, 1:].shape, fb_framey.shape,
        #       fb_frame.shape)
        # print('----END---')
        kd1 = torch.pow(torch.pow(frame[:,:3] + ff_flow[:,:1] * ff_framex + ff_flow[:,1:] * ff_framey - ff_frame, 2).mean(),0.5)
        kd2 = torch.pow(torch.pow(frame[:, 3:] + fb_flow[:, :1] * fb_framex + fb_flow[:, 1:] * fb_framey - fb_frame,2).mean(), 0.5)

        # kd1 = torch.pow(kd1 + 1e-2, 1e-4)
        kd = (kd1 + kd2)/2.



        ff_truth, fb_truth = frame[ :, 3:], frame[:, :3]

        # klloss1 = F.kl_div(ff_frame, ff_truth)
        # klloss2 = F.kl_div(fb_frame, fb_truth)

        ff_tloss = self.tripletloss(ff_truth, ff_frame, fb_frame)
        fb_tloss = self.tripletloss(ff_truth, ff_frame, fb_frame)

        f_ploss, b_ploss = self.log_triplet_loss(ff_truth, ff_frame, fb_frame, 1. - ff_occlu, 1. - fb_occlu)
        total = f_ploss + b_ploss + ff_tloss + fb_tloss +kd
        # total = ff_tloss + fb_tloss

        return total

    def epe(self, source, target):
        with torch.no_grad():
            source = source.cuda()
            target = target.cuda()
            # from termcolor import colored
            # print(colored(f'{source.shape, target.shape, source.max(), target.max()}','red'))
            B, C, H, W = source.size()
            diff = (source - target).reshape(-1, C * H * W)
            return torch.norm(diff)/B
            # return torch.norm(diff, p=2, dim=1).mean().detach()

    def run(self):
        self.initialize()
        for i in range(self.epoch):
            self.train(i)
        self.writer.close()