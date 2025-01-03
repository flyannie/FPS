import wandb
from utils import *
from tqdm import tqdm
from adamp import AdamP
from itertools import cycle
import torch.nn.functional as F
from torch.optim import lr_scheduler

class trainer:
    def __init__(self, source_loader, target_loader, student_model, teacher_model, args):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.args = args
        self.student_model.cuda()
        self.teacher_model.cuda()
        self.loss = torch.nn.L1Loss().cuda()
        self.optimizer = AdamP(self.student_model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 150], gamma=0.1)
        self.curiter = 0
        self.consistency_rampup = 100

    @torch.no_grad()
    def update_teachers(self, teacher, itera, keep_rate=0.996):
        alpha = min(1 - 1 / (itera + 1), keep_rate)
        for ema_param, param in zip(teacher.parameters(), self.student_model.parameters()):
            ema_param.data = (alpha * ema_param.data) + (1 - alpha) * param.data

    def train(self):
        for params in self.teacher_model.parameters():
            params.requires_grad = False
        initialize_weights(self.student_model)
        epoch_iterator = tqdm(range(1, self.args.epoch + 1), ncols=70)
        for epoch in epoch_iterator:
            consistent_loss_avg = 0
            space_loss_avg = 0
            fft_loss_avg = 0
            total_loss_avg = 0
            train_loader = iter(zip(self.target_loader, cycle(self.source_loader)))
            tbar = range(len(self.source_loader))
            tbar = tqdm(tbar, ncols=130, leave=True)
            number = 0
            for i in tbar:
                (real_image, distrub_real_image), (sim_image, distrub_sim_image, t2, adc) = next(train_loader)

                real_image = real_image.cuda(non_blocking=True)
                distrub_real_image = distrub_real_image.cuda(non_blocking=True)
                
                sim_image = sim_image.cuda(non_blocking=True)
                distrub_sim_image = distrub_sim_image.cuda(non_blocking=True)
                t2 = t2.cuda(non_blocking=True)
                adc = adc.cuda(non_blocking=True)

                t2_2 = F.interpolate(t2, scale_factor=0.5, mode='bilinear')
                t2_4 = F.interpolate(t2, scale_factor=0.25, mode='bilinear')
                t2_8 = F.interpolate(t2, scale_factor=0.125, mode='bilinear')
                adc_2 = F.interpolate(adc, scale_factor=0.5, mode='bilinear')
                adc_4 = F.interpolate(adc, scale_factor=0.25, mode='bilinear')
                adc_8 = F.interpolate(adc, scale_factor=0.125, mode='bilinear')
                
                with torch.no_grad():
                    t_s_outs = self.teacher_model(distrub_sim_image)
                    t_r_outs = self.teacher_model(distrub_real_image)
                
                s_s_outs = self.student_model(sim_image)
                s_r_outs = self.student_model(real_image)

                sim_consistent_loss = self.loss(t_s_outs[0], s_s_outs[0]) + self.loss(t_s_outs[1], s_s_outs[1]) + self.loss(t_s_outs[2], s_s_outs[2]) + self.loss(t_s_outs[3], s_s_outs[3])
                real_consistent_loss = self.loss(t_r_outs[0], s_r_outs[0]) +self.loss(t_r_outs[1], s_r_outs[1]) + self.loss(t_r_outs[2], s_r_outs[2]) + self.loss(t_r_outs[3], s_r_outs[3])
                
                weight = self.get_current_consistency_weight(epoch)
                consistent_loss = sim_consistent_loss + weight * real_consistent_loss

                t_space_loss_t2 = self.loss(t_s_outs[0][:,0:1,:,:], t2_8) + self.loss(t_s_outs[1][:,0:1,:,:], t2_4) + self.loss(t_s_outs[2][:,0:1,:,:], t2_2) + self.loss(t_s_outs[3][:,0:1,:,:], t2) 
                t_space_loss_adc = self.loss(t_s_outs[0][:,1:2,:,:], adc_8) + self.loss(t_s_outs[1][:,1:2,:,:], adc_4) + self.loss(t_s_outs[2][:,1:2,:,:], adc_2) + self.loss(t_s_outs[3][:,1:2,:,:], adc) 

                s_space_loss_t2 = self.loss(s_s_outs[0][:,0:1,:,:], t2_8) + self.loss(s_s_outs[1][:,0:1,:,:], t2_4) + self.loss(s_s_outs[2][:,0:1,:,:], t2_2) + self.loss(s_s_outs[3][:,0:1,:,:], t2) 
                s_space_loss_adc = self.loss(s_s_outs[0][:,1:2,:,:], adc_8) + self.loss(s_s_outs[1][:,1:2,:,:], adc_4) + self.loss(s_s_outs[2][:,1:2,:,:], adc_2) + self.loss(s_s_outs[3][:,1:2,:,:], adc) 

                space_loss = t_space_loss_t2 + t_space_loss_adc + s_space_loss_t2 + s_space_loss_adc

                fft_t2_8 = torch.fft.fft2(t2_8, dim=(-2,-1))
                fft_t2_8 = torch.stack((fft_t2_8.real, fft_t2_8.imag), -1)
                t_fft_out_8 = torch.fft.fft2(t_s_outs[0][:,0:1,:,:], dim=(-2,-1))
                t_fft_out_8 = torch.stack((t_fft_out_8.real, t_fft_out_8.imag), -1)
                s_fft_out_8 = torch.fft.fft2(s_s_outs[0][:,0:1,:,:], dim=(-2,-1))
                s_fft_out_8 = torch.stack((s_fft_out_8.real, s_fft_out_8.imag), -1)

                fft_t2_4 = torch.fft.fft2(t2_4, dim=(-2,-1))
                fft_t2_4 = torch.stack((fft_t2_4.real, fft_t2_4.imag), -1)
                t_fft_out_4 = torch.fft.fft2(t_s_outs[1][:,0:1,:,:], dim=(-2,-1))
                t_fft_out_4 = torch.stack((t_fft_out_4.real, t_fft_out_4.imag), -1)
                s_fft_out_4 = torch.fft.fft2(s_s_outs[1][:,0:1,:,:], dim=(-2,-1))
                s_fft_out_4 = torch.stack((s_fft_out_4.real, s_fft_out_4.imag), -1)

                fft_t2_2 = torch.fft.fft2(t2_2, dim=(-2,-1))
                fft_t2_2 = torch.stack((fft_t2_2.real, fft_t2_2.imag), -1)
                t_fft_out_2 = torch.fft.fft2(t_s_outs[2][:,0:1,:,:], dim=(-2,-1))
                t_fft_out_2 = torch.stack((t_fft_out_2.real, t_fft_out_2.imag), -1)
                s_fft_out_2 = torch.fft.fft2(s_s_outs[2][:,0:1,:,:], dim=(-2,-1))
                s_fft_out_2 = torch.stack((s_fft_out_2.real, s_fft_out_2.imag), -1)

                fft_t2 = torch.fft.fft2(t2, dim=(-2,-1))
                fft_t2 = torch.stack((fft_t2.real, fft_t2.imag), -1)
                t_fft_out = torch.fft.fft2(t_s_outs[3][:,0:1,:,:], dim=(-2,-1))
                t_fft_out = torch.stack((t_fft_out.real, t_fft_out.imag), -1)
                s_fft_out = torch.fft.fft2(s_s_outs[3][:,0:1,:,:], dim=(-2,-1))
                s_fft_out = torch.stack((s_fft_out.real, s_fft_out.imag), -1)

                fft_adc_8 = torch.fft.fft2(adc_8, dim=(-2,-1))
                fft_adc_8 = torch.stack((fft_adc_8.real, fft_adc_8.imag), -1)
                t_fft_out_8_adc = torch.fft.fft2(t_s_outs[0][:,1:2,:,:], dim=(-2,-1))
                t_fft_out_8_adc = torch.stack((t_fft_out_8_adc.real, t_fft_out_8_adc.imag), -1)
                s_fft_out_8_adc = torch.fft.fft2(s_s_outs[0][:,1:2,:,:], dim=(-2,-1))
                s_fft_out_8_adc = torch.stack((s_fft_out_8_adc.real, s_fft_out_8_adc.imag), -1)

                fft_adc_4 = torch.fft.fft2(adc_4, dim=(-2,-1))
                fft_adc_4 = torch.stack((fft_adc_4.real, fft_adc_4.imag), -1)
                t_fft_out_4_adc = torch.fft.fft2(t_s_outs[1][:,1:2,:,:], dim=(-2,-1))
                t_fft_out_4_adc = torch.stack((t_fft_out_4_adc.real, t_fft_out_4_adc.imag), -1)
                s_fft_out_4_adc = torch.fft.fft2(s_s_outs[1][:,1:2,:,:], dim=(-2,-1))
                s_fft_out_4_adc = torch.stack((s_fft_out_4_adc.real, s_fft_out_4_adc.imag), -1)

                fft_adc_2 = torch.fft.fft2(adc_2, dim=(-2,-1))
                fft_adc_2 = torch.stack((fft_adc_2.real, fft_adc_2.imag), -1)
                t_fft_out_2_adc = torch.fft.fft2(t_s_outs[2][:,1:2,:,:], dim=(-2,-1))
                t_fft_out_2_adc = torch.stack((t_fft_out_2_adc.real, t_fft_out_2_adc.imag), -1)
                s_fft_out_2_adc = torch.fft.fft2(s_s_outs[2][:,1:2,:,:], dim=(-2,-1))
                s_fft_out_2_adc = torch.stack((s_fft_out_2_adc.real, s_fft_out_2_adc.imag), -1)

                fft_adc = torch.fft.fft2(adc, dim=(-2,-1))
                fft_adc = torch.stack((fft_adc.real, fft_adc.imag), -1)
                t_fft_out_adc = torch.fft.fft2(t_s_outs[3][:,1:2,:,:], dim=(-2,-1))
                t_fft_out_adc = torch.stack((t_fft_out_adc.real, t_fft_out_adc.imag), -1)
                s_fft_out_adc = torch.fft.fft2(s_s_outs[3][:,1:2,:,:], dim=(-2,-1))
                s_fft_out_adc = torch.stack((s_fft_out_adc.real, s_fft_out_adc.imag), -1)

                t_fft_loss_t2 = self.loss(t_fft_out_8 ,fft_t2_8) + self.loss(t_fft_out_4 ,fft_t2_4) + self.loss(t_fft_out_2 ,fft_t2_2) + self.loss(t_fft_out ,fft_t2) 
                s_fft_loss_t2 = self.loss(s_fft_out_8 ,fft_t2_8) + self.loss(s_fft_out_4 ,fft_t2_4) + self.loss(s_fft_out_2 ,fft_t2_2) + self.loss(s_fft_out ,fft_t2) 
                t_fft_loss_adc = self.loss(t_fft_out_8_adc ,fft_adc_8) + self.loss(t_fft_out_4_adc ,fft_adc_4) + self.loss(t_fft_out_2_adc ,fft_adc_2) + self.loss(t_fft_out_adc ,fft_adc) 
                s_fft_loss_adc = self.loss(s_fft_out_8_adc ,fft_adc_8) + self.loss(s_fft_out_4_adc ,fft_adc_4) + self.loss(s_fft_out_2_adc ,fft_adc_2) + self.loss(s_fft_out_adc ,fft_adc) 

                fft_loss = t_fft_loss_t2 + s_fft_loss_t2 + t_fft_loss_adc + s_fft_loss_adc

                consistent_loss_avg += consistent_loss
                space_loss_avg += space_loss
                fft_loss_avg += fft_loss

                total_loss = consistent_loss + space_loss + 0.1 * fft_loss
                total_loss_avg += total_loss
                number += 1

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                tbar.set_description('Epoch {} | cl {:.4f} sl {:.4f} fl {:.4f} Total {:.4f}|'.format(epoch, consistent_loss, space_loss, fft_loss, total_loss))
                wandb.log({'epoch': epoch,'curiter': self.curiter, 'consistent_loss': consistent_loss, 'space_loss': space_loss, 'fft_loss': fft_loss, 'total_loss':total_loss})

                with torch.no_grad():
                    self.update_teachers(teacher=self.teacher_model, itera=self.curiter)
                    self.curiter = self.curiter + 1

            self.lr_scheduler.step(epoch=epoch - 1)

            if epoch % self.args.save_frequency == 0:
                state = {'arch': type(self.student_model).__name__,
                         'epoch': epoch,
                         'state_dict': self.student_model.state_dict(),
                         'optimizer_dict': self.optimizer.state_dict()}
                ckpt_name = str(self.args.save_path) + 'model_e{}.pth'.format(str(epoch))
                print("Saving a checkpoint: {} ...".format(str(ckpt_name)))
                torch.save(state, ckpt_name)

    def get_current_consistency_weight(self, epoch):
        return self.sigmoid_rampup(epoch, 200)

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))