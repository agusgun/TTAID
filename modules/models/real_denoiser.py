import logging
import os
import shutil
import time
import torch

from modules.models.base import BaseModel
from modules.components import build_components

from tensorboardX import SummaryWriter
from utils.plot import plot_samples_per_epoch, plot_image, stitch
from utils.misc import AverageMeter
from tqdm import tqdm
from utils.metrics import calculate_batch_psnr, calculate_batch_ssim
from pathlib import Path


class RealDenoiserBase(BaseModel):
    def __init__(self, args, device):
        super(RealDenoiserBase, self).__init__(args, device)
        
        _, self.restoration_net, self.rec_criterion = build_components(self.args)

        self.optimizer = torch.optim.Adam(self.restoration_net.parameters(), lr=self.args.learning_rate, betas=(self.args.beta1, self.args.beta2))
        
        self.current_iteration = 0
        self.current_epoch = 0
    
        self.move_components_to_device(args.mode)
        
        self.logger = logging.getLogger("RealDenoiserBase Model")

    def load_checkpoint(self, file_path):
        """
        Load checkpoint
        """
        checkpoint = torch.load(file_path)

        self.current_epoch = checkpoint['epoch']
        self.current_iteration = checkpoint['iteration']
        self.restoration_net.load_state_dict(checkpoint['restoration_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info('Chekpoint loaded successfully from {} at epoch: {} and iteration: {}'.format(
            file_path, checkpoint['epoch'], checkpoint['iteration']))
        return self.current_epoch
    
    def save_checkpoint(self, file_name, is_best=0):
        """
        Save checkpoint
        """
        state = {
            'epoch': self.current_epoch + 1, # because epoch is used for loading then this must be added + 1
            'iteration': self.current_iteration, 
            'restoration_net_state_dict': self.restoration_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state, os.path.join(self.args.checkpoint_dir, file_name))

        if is_best:
            shutil.copyfile(os.path.join(self.args.checkpoint_dir, file_name), os.path.join(self.args.checkpoint_dir, 'model_best.pth'))

    def adjust_learning_rate(self, epoch):
        """
        Adjust learning rate every epoch
        """
        pass


    def train_one_epoch(self, train_loader, epoch):
        """
        Training step for each mini-batch
        """
        self.current_epoch = epoch
        self._reset_metric()

        tqdm_batch = tqdm(train_loader, desc='epoch-{}'.format(self.current_epoch))

        self.restoration_net.train()

        end_time = time.time()
        for curr_it, data in enumerate(tqdm_batch):
            self.data_time_meter.update(time.time() - end_time)

            noisy = data['noisy']
            clean = data['clean']

            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.restoration_net.zero_grad()
            self.optimizer.zero_grad()

            out_clean, noisy_rec = self.restoration_net(noisy)

            loss = self.rec_criterion(out_clean, clean) + self.rec_criterion(noisy, noisy_rec)
            loss.backward()

            self.optimizer.step()

            self.loss_meter.update(loss.item())

            self.current_iteration += 1
            self.batch_time_meter.update(time.time() - end_time)
            end_time = time.time()

            self.summary_writer.add_scalar("epoch/loss", self.loss_meter.val, self.current_iteration)

            tqdm_batch.set_description('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f}'.format(
                batch = curr_it + 1,
                size = len(train_loader),
                data = self.data_time_meter.val,
                bt = self.batch_time_meter.val,
                loss = self.loss_meter.val
            ))

        tqdm_batch.close()
        self.logger.info('Training at epoch-{} | LR: {} Loss: {}'.format(str(self.current_epoch), str(self.args.learning_rate), str(self.loss_meter.val)))


    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validation step for each mini-batch
        """
        self.restoration_net.eval()
        if self.args.mode == 'training':
            tqdm_batch = tqdm(val_loader, desc='Validation at epoch-{}'.format(self.current_epoch))
        
            loss_meter = AverageMeter()
            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()

            for curr_it, data in enumerate(tqdm_batch):
                noisy = data['noisy']
                clean = data['clean']

                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                out_clean, _ = self.restoration_net(noisy)

                out_clean = torch.clamp(out_clean, 0., 1.)
                clean = torch.clamp(clean, 0., 1.)
                noisy = torch.clamp(noisy, 0., 1.)

                loss = self.rec_criterion(out_clean, clean)
                psnr_val, bs = calculate_batch_psnr(clean, out_clean)
                ssim_val, _ = calculate_batch_ssim(clean, out_clean)

                loss_meter.update(loss.item())
                psnr_meter.update(psnr_val, bs)
                ssim_meter.update(ssim_val, bs)
                
                tqdm_batch.set_description('({batch}/{size}) Loss: {loss:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}'.format(
                    batch = curr_it + 1,
                    size = len(val_loader),
                    loss = loss_meter.val,
                    psnr = psnr_meter.val,
                    ssim = ssim_meter.val
                ))
                
            
            self.summary_writer.add_scalar("validation/loss", loss_meter.val, self.current_epoch)
            self.summary_writer.add_scalar("validation/psnr", psnr_meter.val, self.current_epoch)
            self.summary_writer.add_scalar("validation/ssim", ssim_meter.val, self.current_epoch)

            # save last image
            out_saved_img = plot_samples_per_epoch(gen_batch=out_clean.data, output_dir=os.path.join(self.args.output_dir, "out"), epoch=self.current_epoch)
            out_saved_img = out_saved_img.transpose((2, 0, 1))

            clean_saved_img = plot_samples_per_epoch(gen_batch=clean.data, output_dir=os.path.join(self.args.output_dir, "clean"), epoch=self.current_epoch)
            clean_saved_img = clean_saved_img.transpose((2, 0, 1))

            noisy_saved_img = plot_samples_per_epoch(gen_batch=noisy.data, output_dir=os.path.join(self.args.output_dir, "noisy"), epoch=self.current_epoch)
            noisy_saved_img = noisy_saved_img.transpose((2, 0, 1))

            self.summary_writer.add_image('validation/out_img', out_saved_img, self.current_epoch)
            self.summary_writer.add_image('validation/clean_img', clean_saved_img, self.current_epoch)
            self.summary_writer.add_image('validation/noisy_img', noisy_saved_img, self.current_epoch)
            
            self.logger.info('Evaluation after epoch-{} | Loss: {} | PSNR: {} | SSIM: {}'.format(
                str(self.current_epoch),
                str(loss_meter.val),
                str(psnr_meter.val),
                str(ssim_meter.val)
            ))
            tqdm_batch.close()
            return psnr_meter.val # to determine is best
        elif self.args.mode == 'validation':
            tqdm_batch = tqdm(val_loader, desc='Validation using full resolution data')

            loss_meter = AverageMeter()
            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()

            for curr_it, data in enumerate(tqdm_batch):
                noisy = data['noisy']
                clean = data['clean']

                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                if self.args.stitch:
                    out_clean = stitch(self.restoration_net, noisy, n_patches=self.args.stitch_n_patches, output_index=0)
                else:
                    out_clean, _ = self.restoration_net(noisy)

                out_clean = torch.clamp(out_clean, 0., 1.)
                clean = torch.clamp(clean, 0., 1.)
                noisy = torch.clamp(noisy, 0., 1.)

                out_saved_img = plot_image(out_clean.data[0], output_dir=os.path.join(self.args.output_dir, "out"), fname='{}.png'.format(curr_it))
                noisy_saved_img = plot_image(noisy.data[0], output_dir=os.path.join(self.args.output_dir, "noisy"), fname='{}.png'.format(curr_it))
                clean_saved_img = plot_image(clean.data[0], output_dir=os.path.join(self.args.output_dir, "clean"), fname='{}.png'.format(curr_it))
                
                psnr_val, bs = calculate_batch_psnr(clean, out_clean)
                ssim_val, _ = calculate_batch_ssim(clean, out_clean)
                
                psnr_meter.update(psnr_val, bs)
                ssim_meter.update(ssim_val, bs)
                
                tqdm_batch.set_description('({batch}/{size}) | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}'.format(
                    batch = curr_it + 1,
                    size = len(val_loader),
                    psnr = psnr_meter.val,
                    ssim = ssim_meter.val
                ))
            
            self.summary_writer.add_scalar("validation/loss", loss_meter.val, self.current_epoch)
            self.summary_writer.add_scalar("validation/psnr", psnr_meter.val, self.current_epoch)
            self.summary_writer.add_scalar("validation/ssim", ssim_meter.val, self.current_epoch)
            
            self.logger.info('Evaluation after epoch-{} | PSNR: {} | SSIM: {}'.format(
                str(self.current_epoch),
                str(psnr_meter.val),
                str(ssim_meter.val)
            ))

            tqdm_batch.close()
        else: # testing (no ground truth)
            tqdm_batch = tqdm(val_loader, desc='Testing using real data')

            loss_meter = AverageMeter()
            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()

            for curr_it, data in enumerate(tqdm_batch):
                noisy = data['noisy']
                clean = data['clean']

                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                if self.args.stitch:
                    out_clean = stitch(self.restoration_net, noisy, n_patches=self.args.stitch_n_patches, output_index=0)
                else:
                    out_clean, _ = self.restoration_net(noisy)

                out_clean = torch.clamp(out_clean, 0., 1.)
                clean = torch.clamp(clean, 0., 1.)
                noisy = torch.clamp(noisy, 0., 1.)

                out_saved_img = plot_image(out_clean.data[0], output_dir=os.path.join(self.args.output_dir, "out"), fname='{}.png'.format(curr_it))
                noisy_saved_img = plot_image(noisy.data[0], output_dir=os.path.join(self.args.output_dir, "noisy"), fname='{}.png'.format(curr_it))
                clean_saved_img = plot_image(clean.data[0], output_dir=os.path.join(self.args.output_dir, "clean"), fname='{}.png'.format(curr_it))
                
                psnr_val, bs = calculate_batch_psnr(clean, out_clean)
                ssim_val, _ = calculate_batch_ssim(clean, out_clean)
                
                psnr_meter.update(psnr_val, bs)
                ssim_meter.update(ssim_val, bs)
                
                tqdm_batch.set_description('({batch}/{size}) | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}'.format(
                    batch = curr_it + 1,
                    size = len(val_loader),
                    psnr = psnr_meter.val,
                    ssim = ssim_meter.val
                ))
            
            self.summary_writer.add_scalar("testing/loss", loss_meter.val, self.current_epoch)
            self.summary_writer.add_scalar("testing/psnr", psnr_meter.val, self.current_epoch)
            self.summary_writer.add_scalar("testing/ssim", ssim_meter.val, self.current_epoch)
            
            self.logger.info('Evaluation after epoch-{} | PSNR: {} | SSIM: {}'.format(
                str(self.current_epoch),
                str(psnr_meter.val),
                str(ssim_meter.val)
            ))

            tqdm_batch.close()

    def init_training_logger(self):
        """
        Initialize training logger specific for each model
        """
        self.summary_writer = SummaryWriter(log_dir=self.args.summary_dir, comment='RealDenoiser Base')
        Path(os.path.join(self.args.output_dir, 'noisy')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.args.output_dir, 'clean')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.args.output_dir, 'out')).mkdir(parents=True, exist_ok=True)
        self._reset_metric()

    def init_validation_logger(self):
        """
        Initialize validation logger specific for each model
        """
        self.summary_writer = SummaryWriter(log_dir=self.args.summary_dir, comment='RealDenoiser Base')
        Path(os.path.join(self.args.output_dir, 'noisy')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.args.output_dir, 'clean')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.args.output_dir, 'out')).mkdir(parents=True, exist_ok=True)

    def init_testing_logger(self):
        """
        Initialize testing logger specific for each model
        """
        self.summary_writer = SummaryWriter(log_dir=self.args.summary_dir, comment='RealDenoiser Base')
        Path(os.path.join(self.args.output_dir, 'noisy')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.args.output_dir, 'clean')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.args.output_dir, 'out')).mkdir(parents=True, exist_ok=True)

    def finalize_training(self):
        """
        Finalize training
        """
        self.logger.info("Finalizing everything")
        self.save_checkpoint("final_checkpoint.pth")
        self.summary_writer.export_scalars_to_json(os.path.join(self.args.summary_dir, "all_scalars.json"))
        self.summary_writer.close()

    def move_components_to_device(self, mode):
        """
        Move components to device
        """
        self.restoration_net = self.restoration_net.to(self.device)

    def _reset_metric(self):
        """
        Metric related to average meter
        """
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.loss_meter = AverageMeter()

    def count_parameters(self):
        """
        Return the number of parameters for the model
        """
        return sum(p.numel() for p in self.restoration_net.parameters() if p.requires_grad)