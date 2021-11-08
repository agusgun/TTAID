import logging
import os
import shutil
import time
import torch


from copy import deepcopy
from collections import OrderedDict
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

        self.mask_net, self.restoration_net, self.rec_criterion = build_components(
            self.args
        )

        self.mask_optimizer = torch.optim.Adam(
            self.mask_net.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
        )
        self.restoration_optimizer = torch.optim.Adam(
            self.restoration_net.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
        )

        self.current_iteration = 0
        self.current_epoch = 0

        self.move_components_to_device(args.mode)

        self.logger = logging.getLogger("RealDenoiserBase Model")

    def load_checkpoint(self, file_path):
        """
        Load checkpoint
        """
        checkpoint = torch.load(file_path)

        self.current_epoch = checkpoint["epoch"]
        self.current_iteration = checkpoint["iteration"]
        self.restoration_net.load_state_dict(checkpoint["restoration_net_state_dict"])
        self.mask_net.load_state_dict(checkpoint["mask_net_state_dict"])
        self.mask_optimizer.load_state_dict(checkpoint["mask_optimizer"])
        self.restoration_optimizer.load_state_dict(checkpoint["restoration_optimizer"])
        self.logger.info(
            "Chekpoint loaded successfully from {} at epoch: {} and iteration: {}".format(
                file_path, checkpoint["epoch"], checkpoint["iteration"]
            )
        )
        return self.current_epoch

    def save_checkpoint(self, file_name, is_best=0):
        """
        Save checkpoint
        """
        state = {
            "epoch": self.current_epoch
            + 1,  # because epoch is used for loading then this must be added + 1
            "iteration": self.current_iteration,
            "restoration_net_state_dict": self.restoration_net.state_dict(),
            "mask_net_state_dict": self.mask_net.state_dict(),
            "mask_optimizer": self.mask_optimizer.state_dict(),
            "restoration_optimizer": self.restoration_optimizer.state_dict(),
        }

        torch.save(state, os.path.join(self.args.checkpoint_dir, file_name))

        if is_best:
            shutil.copyfile(
                os.path.join(self.args.checkpoint_dir, file_name),
                os.path.join(self.args.checkpoint_dir, "model_best.pth"),
            )

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

        tqdm_batch = tqdm(
            train_loader,
            desc="epoch-{} conventional training".format(self.current_epoch),
        )

        self.restoration_net.train()
        self.mask_net.train()

        end_time = time.time()
        for curr_it, data in enumerate(tqdm_batch):
            self.data_time_meter.update(time.time() - end_time)

            noisy = data["noisy"]
            clean = data["clean"]

            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.restoration_optimizer.zero_grad()

            out_clean, noisy_rec = self.restoration_net(noisy)

            mask = self.mask_net(noisy)
            num_non_zero = torch.count_nonzero(mask)
            num_zero = mask.size()[0] * 256 * 256 - num_non_zero

            noisy_rec_loss = torch.mean(torch.abs(noisy - noisy_rec) * mask)
            clean_loss = self.rec_criterion(out_clean, clean)
            loss = clean_loss + noisy_rec_loss
            loss.backward()

            self.restoration_optimizer.step()

            self.clean_loss_meter.update(clean_loss.item())
            self.noisy_loss_meter.update(noisy_rec_loss.item())

            self.current_iteration += 1
            self.batch_time_meter.update(time.time() - end_time)
            end_time = time.time()

            self.summary_writer.add_scalar(
                "epoch/conventional/loss_clean",
                self.clean_loss_meter.val,
                self.current_iteration,
            )
            self.summary_writer.add_scalar(
                "epoch/conventional/loss_noisy",
                self.noisy_loss_meter.val,
                self.current_iteration,
            )

            tqdm_batch.set_description(
                "ConventionalTraining: ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | LossC: {lossC:.4f} | LossN: {lossN:.4f}".format(
                    batch=curr_it + 1,
                    size=len(train_loader),
                    data=self.data_time_meter.val,
                    bt=self.batch_time_meter.val,
                    lossC=self.clean_loss_meter.val,
                    lossN=self.noisy_loss_meter.val,
                )
            )
        self.logger.info(
            "Training at epoch-{} stage: normal training | LR: {} LossC: {} LossN: {}".format(
                str(self.current_epoch),
                str(self.args.learning_rate),
                str(self.clean_loss_meter.val),
                str(self.noisy_loss_meter.val),
            )
        )
        tqdm_batch.close()

        tqdm_batch = tqdm(
            train_loader, desc="epoch-{} meta-training".format(self.current_epoch)
        )

        end_time = time.time()
        for curr_it, data in enumerate(tqdm_batch):
            self.data_time_meter.update(time.time() - end_time)

            noisy = data["noisy"]
            clean = data["clean"]

            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.restoration_optimizer.zero_grad()

            out_clean, noisy_rec = self.restoration_net(noisy)

            mask = self.mask_net(noisy)
            num_non_zero = torch.count_nonzero(mask)
            num_zero = mask.size()[0] * 256 * 256 - num_non_zero

            noisy_rec_loss = torch.mean(torch.abs(noisy - noisy_rec) * mask)
            clean_loss = self.rec_criterion(out_clean, clean)
            loss = clean_loss + noisy_rec_loss

            # current theta_1
            theta1_weights = OrderedDict(
                (name, param)
                for (name, param) in self.restoration_net.named_parameters()
            )

            # second derivative
            grads = torch.autograd.grad(
                loss, self.restoration_net.parameters(), create_graph=True
            )
            params_data = [p.data for p in list(self.restoration_net.parameters())]

            # theta_1+
            alpha = self.args.learning_rate
            theta1_weights = OrderedDict(
                (name, param - alpha * grad)
                for ((name, param), grad, data) in zip(
                    theta1_weights.items(), grads, params_data
                )
            )

            # compute primary loss
            out_clean, noisy_rec = self.restoration_net(noisy, theta1_weights)
            clean_loss = self.rec_criterion(out_clean, clean)


            self.mask_optimizer.zero_grad()
            clean_loss.backward()  # add entropy loss here if want
            self.mask_optimizer.step()

            self.clean_loss_meter.update(clean_loss.item())
            self.noisy_loss_meter.update(noisy_rec_loss.item())

            self.current_iteration += 1
            self.batch_time_meter.update(time.time() - end_time)
            end_time = time.time()

            self.summary_writer.add_scalar(
                "epoch/meta-training/clean_loss",
                self.clean_loss_meter.val,
                self.current_iteration,
            )
            self.summary_writer.add_scalar(
                "epoch/meta-training/noisy_loss",
                self.noisy_loss_meter.val,
                self.current_iteration,
            )

            tqdm_batch.set_description(
                "Meta-training ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | LossC: {lossC:.4f} | LossN: {lossN:.4f}".format(
                    batch=curr_it + 1,
                    size=len(train_loader),
                    data=self.data_time_meter.val,
                    bt=self.batch_time_meter.val,
                    lossC=self.clean_loss_meter.val,
                    lossN=self.noisy_loss_meter.val,
                )
            )
        self.logger.info(
            "Meta-training: Training at epoch-{} stage: normal training | LR: {} LossC: {} LossN: {}".format(
                str(self.current_epoch),
                str(self.args.learning_rate),
                str(self.clean_loss_meter.val),
                str(self.noisy_loss_meter.val),
            )
        )
        tqdm_batch.close()

    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validation step for each mini-batch
        """
        self.restoration_net.eval()
        self.mask_net.eval()
        if self.args.mode == "training":
            tqdm_batch = tqdm(
                val_loader, desc="Validation at epoch-{}".format(self.current_epoch)
            )

            loss_meter = AverageMeter()
            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()

            for curr_it, data in enumerate(tqdm_batch):
                noisy = data["noisy"]
                clean = data["clean"]

                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                out_clean, _ = self.restoration_net(noisy)
                mask = self.mask_net(noisy)

                out_clean = torch.clamp(out_clean, 0.0, 1.0)
                clean = torch.clamp(clean, 0.0, 1.0)
                noisy = torch.clamp(noisy, 0.0, 1.0)

                loss = self.rec_criterion(out_clean, clean)
                psnr_val, bs = calculate_batch_psnr(clean, out_clean)
                ssim_val, _ = calculate_batch_ssim(clean, out_clean)

                loss_meter.update(loss.item())
                psnr_meter.update(psnr_val, bs)
                ssim_meter.update(ssim_val, bs)

                tqdm_batch.set_description(
                    "({batch}/{size}) Loss: {loss:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}".format(
                        batch=curr_it + 1,
                        size=len(val_loader),
                        loss=loss_meter.val,
                        psnr=psnr_meter.val,
                        ssim=ssim_meter.val,
                    )
                )

            self.summary_writer.add_scalar(
                "validation/loss", loss_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "validation/psnr", psnr_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "validation/ssim", ssim_meter.val, self.current_epoch
            )

            # save last image
            out_saved_img = plot_samples_per_epoch(
                gen_batch=out_clean.data,
                output_dir=os.path.join(self.args.output_dir, "out"),
                epoch=self.current_epoch,
            )
            out_saved_img = out_saved_img.transpose((2, 0, 1))

            clean_saved_img = plot_samples_per_epoch(
                gen_batch=clean.data,
                output_dir=os.path.join(self.args.output_dir, "clean"),
                epoch=self.current_epoch,
            )
            clean_saved_img = clean_saved_img.transpose((2, 0, 1))

            noisy_saved_img = plot_samples_per_epoch(
                gen_batch=noisy.data,
                output_dir=os.path.join(self.args.output_dir, "noisy"),
                epoch=self.current_epoch,
            )
            noisy_saved_img = noisy_saved_img.transpose((2, 0, 1))

            mask_saved_img = plot_samples_per_epoch(
                gen_batch=mask.data,
                output_dir=os.path.join(self.args.output_dir, "mask"),
                epoch=self.current_epoch,
            )
            mask_saved_img = mask_saved_img.transpose((2, 0, 1))

            self.summary_writer.add_image(
                "validation/out_img", out_saved_img, self.current_epoch
            )
            self.summary_writer.add_image(
                "validation/clean_img", clean_saved_img, self.current_epoch
            )
            self.summary_writer.add_image(
                "validation/noisy_img", noisy_saved_img, self.current_epoch
            )
            self.summary_writer.add_image(
                "validation/mask_img", mask_saved_img, self.current_epoch
            )

            self.logger.info(
                "Evaluation after epoch-{} | Loss: {} | PSNR: {} | SSIM: {}".format(
                    str(self.current_epoch),
                    str(loss_meter.val),
                    str(psnr_meter.val),
                    str(ssim_meter.val),
                )
            )
            tqdm_batch.close()
            return psnr_meter.val  # to determine is best
        elif self.args.mode == "validation":
            tqdm_batch = tqdm(val_loader, desc="Validation using full resolution data")

            loss_meter = AverageMeter()
            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()

            for curr_it, data in enumerate(tqdm_batch):
                noisy = data["noisy"]
                clean = data["clean"]

                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                if self.args.stitch:
                    out_clean = stitch(
                        self.restoration_net,
                        noisy,
                        n_patches=self.args.stitch_n_patches,
                        output_index=0,
                    )
                else:
                    out_clean, _ = self.restoration_net(noisy)

                out_clean = torch.clamp(out_clean, 0.0, 1.0)
                clean = torch.clamp(clean, 0.0, 1.0)
                noisy = torch.clamp(noisy, 0.0, 1.0)

                out_saved_img = plot_image(
                    out_clean.data[0],
                    output_dir=os.path.join(self.args.output_dir, "out"),
                    fname="{}.png".format(curr_it),
                )
                noisy_saved_img = plot_image(
                    noisy.data[0],
                    output_dir=os.path.join(self.args.output_dir, "noisy"),
                    fname="{}.png".format(curr_it),
                )
                clean_saved_img = plot_image(
                    clean.data[0],
                    output_dir=os.path.join(self.args.output_dir, "clean"),
                    fname="{}.png".format(curr_it),
                )

                psnr_val, bs = calculate_batch_psnr(clean, out_clean)
                ssim_val, _ = calculate_batch_ssim(clean, out_clean)

                psnr_meter.update(psnr_val, bs)
                ssim_meter.update(ssim_val, bs)

                tqdm_batch.set_description(
                    "({batch}/{size}) | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}".format(
                        batch=curr_it + 1,
                        size=len(val_loader),
                        psnr=psnr_meter.val,
                        ssim=ssim_meter.val,
                    )
                )

            self.summary_writer.add_scalar(
                "validation/loss", loss_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "validation/psnr", psnr_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "validation/ssim", ssim_meter.val, self.current_epoch
            )

            self.logger.info(
                "Evaluation after epoch-{} | PSNR: {} | SSIM: {}".format(
                    str(self.current_epoch), str(psnr_meter.val), str(ssim_meter.val)
                )
            )

            tqdm_batch.close()
        else:  # testing (no ground truth)
            tqdm_batch = tqdm(val_loader, desc="Testing using real data")

            loss_meter = AverageMeter()
            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()

            for curr_it, data in enumerate(tqdm_batch):
                noisy = data["noisy"]
                clean = data["clean"]

                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                if self.args.stitch:
                    out_clean = stitch(
                        self.restoration_net,
                        noisy,
                        n_patches=self.args.stitch_n_patches,
                        output_index=0,
                    )
                else:
                    out_clean, _ = self.restoration_net(noisy)

                out_clean = torch.clamp(out_clean, 0.0, 1.0)
                clean = torch.clamp(clean, 0.0, 1.0)
                noisy = torch.clamp(noisy, 0.0, 1.0)

                out_saved_img = plot_image(
                    out_clean.data[0],
                    output_dir=os.path.join(self.args.output_dir, "out"),
                    fname="{}.png".format(curr_it),
                )
                noisy_saved_img = plot_image(
                    noisy.data[0],
                    output_dir=os.path.join(self.args.output_dir, "noisy"),
                    fname="{}.png".format(curr_it),
                )
                clean_saved_img = plot_image(
                    clean.data[0],
                    output_dir=os.path.join(self.args.output_dir, "clean"),
                    fname="{}.png".format(curr_it),
                )

                psnr_val, bs = calculate_batch_psnr(clean, out_clean)
                ssim_val, _ = calculate_batch_ssim(clean, out_clean)

                psnr_meter.update(psnr_val, bs)
                ssim_meter.update(ssim_val, bs)

                tqdm_batch.set_description(
                    "({batch}/{size}) | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}".format(
                        batch=curr_it + 1,
                        size=len(val_loader),
                        psnr=psnr_meter.val,
                        ssim=ssim_meter.val,
                    )
                )

            self.summary_writer.add_scalar(
                "testing/loss", loss_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "testing/psnr", psnr_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "testing/ssim", ssim_meter.val, self.current_epoch
            )

            self.logger.info(
                "Evaluation after epoch-{} | PSNR: {} | SSIM: {}".format(
                    str(self.current_epoch), str(psnr_meter.val), str(ssim_meter.val)
                )
            )

            tqdm_batch.close()

    def init_training_logger(self):
        """
        Initialize training logger specific for each model
        """
        self.summary_writer = SummaryWriter(
            log_dir=self.args.summary_dir, comment="RealDenoiser Base"
        )
        Path(os.path.join(self.args.output_dir, "noisy")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "clean")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "out")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "mask")).mkdir(
            parents=True, exist_ok=True
        )
        self._reset_metric()

    def init_validation_logger(self):
        """
        Initialize validation logger specific for each model
        """
        self.summary_writer = SummaryWriter(
            log_dir=self.args.summary_dir, comment="RealDenoiser Base"
        )
        Path(os.path.join(self.args.output_dir, "noisy")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "clean")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "out")).mkdir(
            parents=True, exist_ok=True
        )

    def init_testing_logger(self):
        """
        Initialize testing logger specific for each model
        """
        self.summary_writer = SummaryWriter(
            log_dir=self.args.summary_dir, comment="RealDenoiser Base"
        )
        Path(os.path.join(self.args.output_dir, "noisy")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "clean")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "out")).mkdir(
            parents=True, exist_ok=True
        )

    def finalize_training(self):
        """
        Finalize training
        """
        self.logger.info("Finalizing everything")
        self.save_checkpoint("final_checkpoint.pth")
        self.summary_writer.export_scalars_to_json(
            os.path.join(self.args.summary_dir, "all_scalars.json")
        )
        self.summary_writer.close()

    def move_components_to_device(self, mode):
        """
        Move components to device
        """
        self.restoration_net = self.restoration_net.to(self.device)
        self.mask_net = self.mask_net.to(self.device)

    def _reset_metric(self):
        """
        Metric related to average meter
        """
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.clean_loss_meter = AverageMeter()
        self.noisy_loss_meter = AverageMeter()

    def count_parameters(self):
        """
        Return the number of parameters for the model
        """
        num_params_restoration = sum(
            p.numel() for p in self.restoration_net.parameters() if p.requires_grad
        )
        num_params_mask = sum(
            p.numel() for p in self.mask_net.parameters() if p.requires_grad
        )
        return num_params_restoration + num_params_mask


class RealDenoiserMetaTransfer(BaseModel):
    def __init__(self, args, device):
        super(RealDenoiserMetaTransfer, self).__init__(args, device)
        self.logger = logging.getLogger("RealDenoiserBase Model")

        self.mask_net, self.restoration_net, self.rec_criterion = build_components(
            self.args
        )
        self._load_pretrained_model(args.pretrained_path)

        self.mask_optimizer = torch.optim.Adam( # Optim for Mask Net
            self.mask_net.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
        )
        self.restoration_optimizer = torch.optim.Adam( # Meta Optim
            self.restoration_net.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
        )

        self.current_iteration = 0
        self.current_epoch = 0

        self.move_components_to_device(args.mode)

        # Meta Learning
        self.num_inner_loop = 5 # TODO: inner_lr / outer_lr
        self.inner_lr = args.learning_rate / 4 # 0.00001 works previously

        self.weight_name = [name for name, _ in self.restoration_net.named_parameters() if 'head' in name]
        self.weight_len = len(self.weight_name)
        torch.autograd.set_detect_anomaly(True)

    def _load_pretrained_model(self, pretrained_path):
        if not pretrained_path is None:
            checkpoint = torch.load(pretrained_path)
            self.restoration_net.load_state_dict(checkpoint["restoration_net_state_dict"])
            self.mask_net.load_state_dict(checkpoint["mask_net_state_dict"])
            self.logger.info("Succesfully loaded from {}".format(pretrained_path))

    def load_checkpoint(self, file_path):
        """
        Load checkpoint
        """
        checkpoint = torch.load(file_path)

        self.current_epoch = checkpoint["epoch"]
        self.current_iteration = checkpoint["iteration"]
        self.restoration_net.load_state_dict(checkpoint["restoration_net_state_dict"])
        self.mask_net.load_state_dict(checkpoint["mask_net_state_dict"])
        self.mask_optimizer.load_state_dict(checkpoint["mask_optimizer"])
        self.restoration_optimizer.load_state_dict(checkpoint["restoration_optimizer"])
        self.logger.info(
            "Chekpoint loaded successfully from {} at epoch: {} and iteration: {}".format(
                file_path, checkpoint["epoch"], checkpoint["iteration"]
            )
        )
        return self.current_epoch

    def save_checkpoint(self, file_name, is_best=0):
        """
        Save checkpoint
        """
        state = {
            "epoch": self.current_epoch
            + 1,  # because epoch is used for loading then this must be added + 1
            "iteration": self.current_iteration,
            "restoration_net_state_dict": self.restoration_net.state_dict(),
            "mask_net_state_dict": self.mask_net.state_dict(),
            "mask_optimizer": self.mask_optimizer.state_dict(),
            "restoration_optimizer": self.restoration_optimizer.state_dict(),
        }

        torch.save(state, os.path.join(self.args.checkpoint_dir, file_name))

        if is_best:
            shutil.copyfile(
                os.path.join(self.args.checkpoint_dir, file_name),
                os.path.join(self.args.checkpoint_dir, "model_best.pth"),
            )

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

        self.restoration_net.train()
        self.mask_net.train()

        end_time = time.time()

        tqdm_batch = tqdm(
            train_loader,
            desc="epoch-{} meta transfer learning".format(self.current_epoch)
        )
        for curr_it, data in enumerate(tqdm_batch):
            self.data_time_meter.update(time.time() - end_time)

            noisy_data = data["noisy"].to(self.device)
            clean_data = data["clean"].to(self.device)

            # Optimization
            self.restoration_optimizer.zero_grad()
            self.mask_optimizer.zero_grad()

            # Freeze network body
            for name, param in self.restoration_net.named_parameters():
                if "head" in name:
                    break
                param.require_grad = False

            outer_loss = 0
            for iter_batch in range(noisy_data.size()[0]):
                noisy, clean = noisy_data[iter_batch].unsqueeze(0), clean_data[iter_batch].unsqueeze(0)

                ### Inner Loop Start Per Sample
                for iter_loop in range(self.num_inner_loop):
                    ## Aux Head

                    # ## TODO: find a way to update auxiliary head or put this outside iter loop
                    # _, noisy_rec = self.restoration_net(noisy)

                    # mask = self.mask_net(noisy)
                    # mask = (mask > 0.5).float()
                    # num_non_zero = torch.count_nonzero(mask)
                    # num_zero = mask.size()[0] * 256 * 256 - num_non_zero

                    # noisy_rec_loss = torch.sum(
                    #     (torch.abs((noisy_rec - noisy) * mask)) / torch.sum(mask)
                    # )

                    # loss_for_inner_update = noisy_rec_loss

                    # ## Auxiliary Head Update
                    # for name, param in self.restoration_net.primary_head.named_parameters():
                    #     param.requires_grad = False

                    # self.restoration_optimizer.zero_grad()
                    # loss_for_inner_update.backward()
                    # self.restoration_optimizer.step()

                    # for name, param in self.restoration_net.primary_head.named_parameters():
                    #     param.requires_grad = True

                    ## Adapted Parameter
                    _, noisy_rec = self.restoration_net(noisy)

                    mask = self.mask_net(noisy)
                    mask = (mask > 0.5).float()
                    num_non_zero = torch.count_nonzero(mask)
                    num_zero = mask.size()[0] * 256 * 256 - num_non_zero

                    noisy_rec_loss = torch.sum(
                        (torch.abs((noisy_rec - noisy) * mask)) / (torch.sum(mask) + 1e-8)
                    )

                    # theta_1
                    theta1_weights = OrderedDict(
                        (name, param)
                        for (name, param) in self.restoration_net.named_parameters()
                    )

                    # Second grad for primary head
                    prim_head_grads = torch.autograd.grad(
                        noisy_rec_loss,
                        list(self.restoration_net.parameters())[-9:],
                        create_graph=True
                    )

                    # theta_1+
                    theta1_new_weights = OrderedDict()
                    iter_meta_grads = 0
                    for name, param in theta1_weights.items():
                        if name in self.weight_name:
                            theta1_new_weights[name] = param - self.inner_lr * prim_head_grads[iter_meta_grads]
                            iter_meta_grads += 1
                        else:
                            theta1_new_weights[name] = param
                    
                    ## Logging
                    self.inner_loss_meter.update(noisy_rec_loss.item())

                    # For Outer Loss
                    out_clean, _ = self.restoration_net(noisy, theta1_new_weights)
                    current_outer_loss = self.rec_criterion(out_clean, clean)
                ### Inner Loop End
                outer_loss += current_outer_loss

            ### Outer Loop Start
            for name, param in self.restoration_net.named_parameters():
                param.require_grad = True
            outer_loss = outer_loss / noisy_data.size()[0]
            self.restoration_optimizer.zero_grad()
            outer_loss.backward()
            self.restoration_optimizer.step()
            self.mask_optimizer.step()
            ### Outer Loop End

            ## Logging
            self.outer_loss_meter.update(outer_loss.item())

            self.current_iteration += 1
            self.batch_time_meter.update(time.time() - end_time)
            end_time = time.time()
        
            self.summary_writer.add_scalar(
                "epoch/mtl/inner_loss",
                self.inner_loss_meter.val,
                self.current_iteration,
            )
            self.summary_writer.add_scalar(
                "epoch/mtl/outer_loss",
                self.outer_loss_meter.val,
                self.current_iteration,
            )
            
            tqdm_batch.set_description(
                "MetaTransfeLearning: ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f} | InnerLoss: {innerLoss:.4f} | OuterLoss: {outerLoss:.4f}".format(
                    batch=curr_it + 1,
                    size=len(train_loader),
                    data=self.data_time_meter.val,
                    bt=self.batch_time_meter.val,
                    innerLoss=self.inner_loss_meter.val,
                    outerLoss=self.outer_loss_meter.val,
                )
            )

        self.logger.info(
            "Training at epoch-{} | LR: {} | OuterLoss: {} | InnerLoss: {}".format(
                self.current_epoch, self.args.learning_rate, self.outer_loss_meter.val, self.inner_loss_meter.val
            )
        )
        tqdm_batch.close()

    def validate(self, val_loader):
        """
        Validation step for each mini-batch
        """
        self.restoration_net.eval()
        self.mask_net.eval()
        if self.args.mode == "training":
            tqdm_batch = tqdm(
                val_loader, desc="Validation at epoch-{}".format(self.current_epoch)
            )

            loss_meter = AverageMeter()
            psnr_ba_meter = AverageMeter()
            ssim_ba_meter = AverageMeter()
            psnr_aa_meter = AverageMeter()
            ssim_aa_meter = AverageMeter()

            for curr_it, data in enumerate(tqdm_batch):
                noisy = data["noisy"]
                clean = data["clean"]

                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                out_ba = torch.ones_like(clean)

                out = torch.ones_like(clean)
                mask = torch.ones(size=(noisy.size()[0], 1, noisy.size()[2], noisy.size()[3]), dtype=out.dtype, device=out.device)

                for iter_batch in range(noisy.size()[0]):
                    adapted_restoration_net = deepcopy(self.restoration_net)
                    adapted_restoration_net.train()

                    optimizer = torch.optim.Adam(
                        adapted_restoration_net.parameters(),
                        lr=self.args.learning_rate,
                        betas=(self.args.beta1, self.args.beta2),
                    )

                    for name, param in adapted_restoration_net.named_parameters():
                        param.requires_grad = False
                        if 'head' in name:
                            break

                    current_noisy = noisy[iter_batch].unsqueeze(0)
                    current_clean = clean[iter_batch].unsqueeze(0)

                    for i in range(self.num_inner_loop):
                        current_out_clean, current_noisy_rec = adapted_restoration_net(current_noisy)
                        if i == 0:
                            out_ba[iter_batch, :, :, :] = current_out_clean.detach()

                        current_mask = self.mask_net(current_noisy)
                        current_mask = (current_mask > 0.5).float()
                        num_non_zero = torch.count_nonzero(current_mask)
                        num_zero = current_mask.size()[0] * 256 * 256 - num_non_zero

                        aux_loss = torch.sum(
                            (torch.abs((current_noisy_rec - current_noisy) * current_mask)) / torch.sum(current_mask)
                        )
                        optimizer.zero_grad()
                        aux_loss.backward()
                        optimizer.step()

                        current_out_clean = torch.clamp(current_out_clean, 0.0, 1.0)
                    out[iter_batch, :, :, :] = current_out_clean.detach()
                    mask[iter_batch, :, :, :] = current_mask.detach()

                psnr_val, bs = calculate_batch_psnr(clean, out_ba)
                ssim_val, _ = calculate_batch_ssim(clean, out_ba)
                psnr_ba_meter.update(psnr_val, bs)
                ssim_ba_meter.update(ssim_val, bs)

                psnr_val, bs = calculate_batch_psnr(clean, out)
                ssim_val, _ = calculate_batch_ssim(clean, out)

                loss = self.rec_criterion(out, clean)
                loss_meter.update(loss.item())
                psnr_aa_meter.update(psnr_val, bs)
                ssim_aa_meter.update(ssim_val, bs)
                
                tqdm_batch.set_description(
                    "({batch}/{size}) Loss: {loss:.4f} | PSNRBA: {psnrba:.4f} | SSIMBA: {ssimba:.4f} | PSNRAA: {psnraa:.4f} | SSIMAA: {ssimaa:.4f}".format(
                        batch=curr_it + 1,
                        size=len(val_loader),
                        loss=loss_meter.val,
                        psnrba=psnr_ba_meter.val,
                        ssimba=ssim_ba_meter.val,
                        psnraa=psnr_aa_meter.val,
                        ssimaa=ssim_aa_meter.val,
                    )
                )

            out_saved_img = plot_samples_per_epoch(
                out.data,
                output_dir=os.path.join(self.args.output_dir, "out"),
                epoch=self.current_epoch
            )
            noisy_saved_img = plot_samples_per_epoch(
                noisy.data,
                output_dir=os.path.join(self.args.output_dir, "noisy"),
                epoch=self.current_epoch
            )
            clean_saved_img = plot_samples_per_epoch(
                clean.data,
                output_dir=os.path.join(self.args.output_dir, "clean"),
                epoch=self.current_epoch
            )
            mask_saved_img = plot_samples_per_epoch(
                mask.data,
                output_dir=os.path.join(self.args.output_dir, "mask"),
                epoch=self.current_epoch
            )

            self.summary_writer.add_scalar(
                "validation/loss", loss_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "validation/psnr_ba", psnr_ba_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "validation/ssim_ba", ssim_ba_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "validation/psnr_aa", psnr_aa_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "validation/ssim_aa", ssim_aa_meter.val, self.current_epoch
            )

            self.logger.info(
                "Evaluation after epoch-{} | Loss: {} | PSNRBA: {} | SSIMBA: {} | PSNRAA: {} | SSIMAA: {}".format(
                    str(self.current_epoch),
                    str(loss_meter.val),
                    str(psnr_ba_meter.val),
                    str(ssim_ba_meter.val),
                    str(psnr_aa_meter.val),
                    str(ssim_aa_meter.val),
                )
            )
            tqdm_batch.close()
            return psnr_aa_meter.val  # to determine is best
        elif self.args.mode == "validation": # TODO: check
            tqdm_batch = tqdm(
                val_loader, desc="Validation at epoch-{}".format(self.current_epoch)
            )

            loss_meter = AverageMeter()
            psnr_ba_meter = AverageMeter()
            ssim_ba_meter = AverageMeter()
            psnr_aa_meter = AverageMeter()
            ssim_aa_meter = AverageMeter()

            for curr_it, data in enumerate(tqdm_batch):
                noisy = data["noisy"]
                clean = data["clean"]

                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                out_ba = torch.ones_like(clean)

                out = torch.ones_like(clean)
                mask = torch.ones(size=(noisy.size()[0], 1, noisy.size()[2], noisy.size()[3]), dtype=out.dtype, device=out.device)
                for iter_batch in range(noisy.size()[0]):
                    adapted_restoration_net = deepcopy(self.restoration_net)
                    adapted_restoration_net.train()

                    optimizer = torch.optim.Adam(
                        adapted_restoration_net.parameters(),
                        lr=self.args.learning_rate,
                        betas=(self.args.beta1, self.args.beta2),
                    )

                    for name, param in adapted_restoration_net.named_parameters():
                        param.requires_grad = False
                        if 'head' in name:
                            break

                    current_noisy = noisy[iter_batch].unsqueeze(0)
                    current_clean = clean[iter_batch].unsqueeze(0)

                    for i in range(self.num_inner_loop):
                        current_out_clean, current_noisy_rec = adapted_restoration_net(current_noisy)
                        if i == 0:
                            out_ba[iter_batch, :, :, :] = current_out_clean.detach()

                        current_mask = self.mask_net(current_noisy)
                        current_mask = (current_mask > 0.5).float()
                        num_non_zero = torch.count_nonzero(current_mask)
                        num_zero = current_mask.size()[0] * 256 * 256 - num_non_zero

                        aux_loss = torch.sum(
                            (torch.abs((current_noisy_rec - current_noisy) * current_mask)) / torch.sum(current_mask)
                        )
                        optimizer.zero_grad()
                        aux_loss.backward()
                        optimizer.step()

                        current_out_clean = torch.clamp(current_out_clean, 0.0, 1.0)
                    out[iter_batch, :, :, :] = current_out_clean.detach()
                    mask[iter_batch, :, :, :] = current_mask.detach()

                psnr_val, bs = calculate_batch_psnr(clean, out_ba)
                ssim_val, _ = calculate_batch_ssim(clean, out_ba)
                psnr_ba_meter.update(psnr_val, bs)
                ssim_ba_meter.update(ssim_val, bs)

                psnr_val, bs = calculate_batch_psnr(clean, out)
                ssim_val, _ = calculate_batch_ssim(clean, out)

                loss = self.rec_criterion(out, clean)
                loss_meter.update(loss.item())
                psnr_aa_meter.update(psnr_val, bs)
                ssim_aa_meter.update(ssim_val, bs)
                
                out_saved_img = plot_image(
                    out.data[0],
                    output_dir=os.path.join(self.args.output_dir, "out"),
                    fname="{}.png".format(curr_it),
                )
                noisy_saved_img = plot_image(
                    noisy.data[0],
                    output_dir=os.path.join(self.args.output_dir, "noisy"),
                    fname="{}.png".format(curr_it),
                )
                clean_saved_img = plot_image(
                    clean.data[0],
                    output_dir=os.path.join(self.args.output_dir, "clean"),
                    fname="{}.png".format(curr_it),
                )

                tqdm_batch.set_description(
                    "({batch}/{size}) Loss: {loss:.4f} | PSNRBA: {psnrba:.4f} | SSIMBA: {ssimba:.4f} | PSNRAA: {psnraa:.4f} | SSIMAA: {ssimaa:.4f}".format(
                        batch=curr_it + 1,
                        size=len(val_loader),
                        loss=loss_meter.val,
                        psnrba=psnr_ba_meter.val,
                        ssimba=ssim_ba_meter.val,
                        psnraa=psnr_aa_meter.val,
                        ssimaa=ssim_aa_meter.val,
                    )
                )

            self.summary_writer.add_scalar(
                "validation/loss", loss_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "validation/psnr_ba", psnr_ba_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "validation/ssim_ba", ssim_ba_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "validation/psnr_aa", psnr_aa_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "validation/ssim_aa", ssim_aa_meter.val, self.current_epoch
            )

            self.logger.info(
                "Evaluation after epoch-{} | Loss: {} | PSNRBA: {} | SSIMBA: {} | PSNRAA: {} | SSIMAA: {}".format(
                    str(self.current_epoch),
                    str(loss_meter.val),
                    str(psnr_ba_meter.val),
                    str(ssim_ba_meter.val),
                    str(psnr_aa_meter.val),
                    str(ssim_aa_meter.val),
                )
            )
            tqdm_batch.close()
            return psnr_aa_meter.val  # to determine is best
        else:  # testing (no ground truth) # TODO: check
            tqdm_batch = tqdm(
                val_loader, desc="Validation at epoch-{}".format(self.current_epoch)
            )

            loss_meter = AverageMeter()
            psnr_ba_meter = AverageMeter()
            ssim_ba_meter = AverageMeter()
            psnr_aa_meter = AverageMeter()
            ssim_aa_meter = AverageMeter()

            for curr_it, data in enumerate(tqdm_batch):
                noisy = data["noisy"]
                clean = data["clean"]

                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                out_ba = torch.ones_like(clean)

                out = torch.ones_like(clean)
                mask = torch.ones(size=(noisy.size()[0], 1, noisy.size()[2], noisy.size()[3]), dtype=out.dtype, device=out.device)

                for iter_batch in range(noisy.size()[0]):
                    adapted_restoration_net = deepcopy(self.restoration_net)
                    adapted_restoration_net.train()

                    optimizer = torch.optim.Adam(
                        adapted_restoration_net.parameters(),
                        lr=self.args.learning_rate,
                        betas=(self.args.beta1, self.args.beta2),
                    )

                    for name, param in adapted_restoration_net.named_parameters():
                        param.requires_grad = False
                        if 'head' in name:
                            break

                    current_noisy = noisy[iter_batch].unsqueeze(0)
                    current_clean = clean[iter_batch].unsqueeze(0)

                    for i in range(self.num_inner_loop):
                        current_out_clean, current_noisy_rec = adapted_restoration_net(current_noisy)
                        if i == 0:
                            out_ba[iter_batch, :, :, :] = current_out_clean.detach()

                        current_mask = self.mask_net(current_noisy)
                        current_mask = (current_mask > 0.5).float()
                        num_non_zero = torch.count_nonzero(current_mask)
                        num_zero = current_mask.size()[0] * 256 * 256 - num_non_zero

                        aux_loss = torch.sum(
                            (torch.abs((current_noisy_rec - current_noisy) * current_mask)) / torch.sum(current_mask)
                        )
                        optimizer.zero_grad()
                        aux_loss.backward()
                        optimizer.step()

                        current_out_clean = torch.clamp(current_out_clean, 0.0, 1.0)
                    out[iter_batch, :, :, :] = current_out_clean.detach()
                    mask[iter_batch, :, :, :] = current_mask.detach()

                psnr_val, bs = calculate_batch_psnr(clean, out_ba)
                ssim_val, _ = calculate_batch_ssim(clean, out_ba)
                psnr_ba_meter.update(psnr_val, bs)
                ssim_ba_meter.update(ssim_val, bs)

                psnr_val, bs = calculate_batch_psnr(clean, out)
                ssim_val, _ = calculate_batch_ssim(clean, out)

                loss = self.rec_criterion(out, clean)
                loss_meter.update(loss.item())
                psnr_aa_meter.update(psnr_val, bs)
                ssim_aa_meter.update(ssim_val, bs)

                out_saved_img = plot_image(
                    out.data[0],
                    output_dir=os.path.join(self.args.output_dir, "out"),
                    fname="{}.png".format(curr_it),
                )
                noisy_saved_img = plot_image(
                    noisy.data[0],
                    output_dir=os.path.join(self.args.output_dir, "noisy"),
                    fname="{}.png".format(curr_it),
                )
                clean_saved_img = plot_image(
                    clean.data[0],
                    output_dir=os.path.join(self.args.output_dir, "clean"),
                    fname="{}.png".format(curr_it),
                )

                tqdm_batch.set_description(
                    "({batch}/{size}) Loss: {loss:.4f} | PSNRBA: {psnrba:.4f} | SSIMBA: {ssimba:.4f} | PSNRAA: {psnraa:.4f} | SSIMAA: {ssimaa:.4f}".format(
                        batch=curr_it + 1,
                        size=len(val_loader),
                        loss=loss_meter.val,
                        psnrba=psnr_ba_meter.val,
                        ssimba=ssim_ba_meter.val,
                        psnraa=psnr_aa_meter.val,
                        ssimaa=ssim_aa_meter.val,
                    )
                )

            self.summary_writer.add_scalar(
                "testing/loss", loss_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "testing/psnr_ba", psnr_ba_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "testing/ssim_ba", ssim_ba_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "testing/psnr_aa", psnr_aa_meter.val, self.current_epoch
            )
            self.summary_writer.add_scalar(
                "testing/ssim_aa", ssim_aa_meter.val, self.current_epoch
            )

            self.logger.info(
                "Evaluation after epoch-{} | Loss: {} | PSNRBA: {} | SSIMBA: {} | PSNRAA: {} | SSIMAA: {}".format(
                    str(self.current_epoch),
                    str(loss_meter.val),
                    str(psnr_ba_meter.val),
                    str(ssim_ba_meter.val),
                    str(psnr_aa_meter.val),
                    str(ssim_aa_meter.val),
                )
            )
            tqdm_batch.close()
            return psnr_aa_meter.val  # to determine is best

    def init_training_logger(self):
        """
        Initialize training logger specific for each model
        """
        self.summary_writer = SummaryWriter(
            log_dir=self.args.summary_dir, comment="RealDenoiser Base"
        )
        Path(os.path.join(self.args.output_dir, "noisy")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "clean")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "out")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "mask")).mkdir(
            parents=True, exist_ok=True
        )
        self._reset_metric()

    def init_validation_logger(self):
        """
        Initialize validation logger specific for each model
        """
        self.summary_writer = SummaryWriter(
            log_dir=self.args.summary_dir, comment="RealDenoiser Base"
        )
        Path(os.path.join(self.args.output_dir, "noisy")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "clean")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "out")).mkdir(
            parents=True, exist_ok=True
        )

    def init_testing_logger(self):
        """
        Initialize testing logger specific for each model
        """
        self.summary_writer = SummaryWriter(
            log_dir=self.args.summary_dir, comment="RealDenoiser Base"
        )
        Path(os.path.join(self.args.output_dir, "noisy")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "clean")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.args.output_dir, "out")).mkdir(
            parents=True, exist_ok=True
        )

    def finalize_training(self):
        """
        Finalize training
        """
        self.logger.info("Finalizing everything")
        self.save_checkpoint("final_checkpoint.pth")
        self.summary_writer.export_scalars_to_json(
            os.path.join(self.args.summary_dir, "all_scalars.json")
        )
        self.summary_writer.close()

    def move_components_to_device(self, mode):
        """
        Move components to device
        """
        self.restoration_net = self.restoration_net.to(self.device)
        self.mask_net = self.mask_net.to(self.device)

    def _reset_metric(self):
        """
        Metric related to average meter
        """
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.outer_loss_meter = AverageMeter()
        self.inner_loss_meter = AverageMeter()

    def count_parameters(self):
        """
        Return the number of parameters for the model
        """
        num_params_restoration = sum(
            p.numel() for p in self.restoration_net.parameters() if p.requires_grad
        )
        num_params_mask = sum(
            p.numel() for p in self.mask_net.parameters() if p.requires_grad
        )
        return num_params_restoration + num_params_mask