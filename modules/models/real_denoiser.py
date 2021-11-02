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
            self.mask_optimizer.zero_grad()

            out_clean, noisy_rec = self.restoration_net(noisy)
            mask = self.mask_net(noisy)

            mask = (mask > 0.5).float()
            num_non_zero = torch.count_nonzero(mask)
            num_zero = mask.size()[0] * 256 * 256 - num_non_zero

            noisy_rec_loss = torch.sum(
                (torch.abs((noisy_rec - noisy) * mask)) / torch.sum(mask)
            )
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
            self.mask_optimizer.zero_grad()

            out_clean, noisy_rec = self.restoration_net(noisy)
            mask = self.mask_net(noisy)

            mask = (mask > 0.5).float()
            num_non_zero = torch.count_nonzero(mask)
            num_zero = mask.size()[0] * 256 * 256 - num_non_zero

            noisy_rec_loss = torch.sum(
                (torch.abs((noisy_rec - noisy) * mask)) / torch.sum(mask)
            )
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
            out_clean, noisy_rec = self.restoration_net(noisy)
            clean_loss = self.rec_criterion(out_clean, clean)

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

        self.mask_net, self.restoration_net, self.rec_criterion = build_components(
            self.args
        )

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

        self.logger = logging.getLogger("RealDenoiserBase Model")

        # Meta Learning
        self.num_inner_loop = 5 # TODO: inner_lr / outer_lr
        self.inner_lr = args.learning_rate

        self.weight_name = [name for name, _ in self.restoration_net.named_parameters() if 'head' in name]
        for name, param in self.restoration_net.named_parameters():
            print(name)
        self.weight_len = len(self.weight_name)
        self._initialize_parameters() # TODO: use MAXL model

    def _add_default_weights(self, weights): # initialize
        for i, name in enumerate(self.weight_for_default_names):
            weights[name] = self.weight_for_default[i]
        return weights

    def _free_state(self): # initialize
        self.updated_state_dict = OrderedDict()
        self.updated_state_dict = self._add_default_weights(self.updated_state_dict)

    def _initialize_parameters(self): # initialize
        self._store_state()
        self.weight_for_default = torch.nn.ParameterList([])
        self.weight_for_default_names = []
        for name, value in self.keep_weight.items():
            if not name in self.weight_name:
                self.weight_for_default_names.append(name)
                self.weight_for_default.append(
                    torch.nn.Parameter(value.to(dtype=torch.float))
                )
        self._free_state()

    def _store_state(self): # initialize + inner loop
        self.keep_weight = deepcopy(self.restoration_net.state_dict())

    def _load_weights(self): # inner loop
        tmp = deepcopy(self.restoration_net.state_dict())
        weights = []
        for name, value in tmp.items():
            if name in self.weight_name:
                weights.append(value)
        return weights

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

    def train_one_epoch(self, train_loaders, epoch):
        """
        Training step for each mini-batch
        """
        self.current_epoch = epoch
        self._reset_metric()
        self.restoration_net.train()
        self.mask_net.train()

        end_time = time.time()

        self.restoration_optimizer.zero_grad()
        self.mask_optimizer.zero_grad()

        for curr_dt, train_loader in enumerate(train_loaders):
            tqdm_batch = tqdm(
                train_loader,
                desc="database-{0} training, epoch: {1}".format(
                    curr_dt, epoch
                ),
            )
            query_loader = iter(train_loader)

            for curr_it, data in enumerate(tqdm_batch):
                self.data_time_meter.update(time.time() - end_time)

                # Freeze network body
                for name, param in self.restoration_net.named_parameters():
                    if "head" in name:
                        break
                    param.require_grad = False
                
                # Start TODO: random samples
                noisy_support = data["noisy"].to(self.device)
                clean_support = data["clean"].to(self.device)
                
                query = next(query_loader)
                noisy_query = query["noisy"].to(self.device)
                clean_query = query["clean"].to(self.device)

                outer_loss = 0
                print(query["img_path"])
                ### Inner Loop Start TODO: per sample maybe
                for iter_loop in range(self.num_inner_loop):
                    if iter_loop > 0:
                        self.restoration_net.load_state_dict(self.updated_state_dict)
                    ## Adapted Parameter
                    weights_for_autograd = self._load_weights()[:5] # only prim head

                    _, noisy_rec = self.restoration_net(noisy_support)

                    mask = self.mask_net(noisy_support)
                    mask = (mask > 0.5).float()
                    num_non_zero = torch.count_nonzero(mask)
                    num_zero = mask.size()[0] * 256 * 256 - num_non_zero

                    noisy_rec_loss = torch.sum(
                        (torch.abs((noisy_rec - noisy_support) * mask)) / torch.sum(mask)
                    )

                    loss_for_inner_update = noisy_rec_loss

                    grad = torch.autograd.grad(
                        loss_for_inner_update,
                        list(self.restoration_net.parameters())[-9:-4],
                        create_graph=True
                    )
                    for w_idx in range(self.weight_len - 4):
                        self.updated_state_dict[self.weight_name[w_idx]] = weights_for_autograd[w_idx] - self.inner_lr * grad[w_idx]
                    
                    ## Auxiliary Head Update
                    for name, param in self.restoration_net.primary_head.named_parameters():
                        param.requires_grad = False

                    self.restoration_optimizer.zero_grad()
                    loss_for_inner_update.backward()
                    self.restoration_optimizer.step()

                    for w_idx in range(self.weight_len - 4, self.weight_len):
                        self.updated_state_dict[self.weight_name[w_idx]] = self.restoration_net.state_dict()[self.weight_name[w_idx]]

                    for name, param in self.restoration_net.primary_head.named_parameters():
                        param.requires_grad = True

                self.restoration_net.load_state_dict(self.updated_state_dict)
                self.restoration_net.load_state_dict(self.keep_weight)
                query_clean, _ = self.restoration_net(noisy_query)

                ### Inner Loop End
                outer_loss = self.rec_criterion(query_clean, clean_query)

                ### Outer Loop Start
                for name, param in self.restoration_net.named_parameters():
                    param.require_grad = True
                outer_loss = outer_loss
                self.restoration_optimizer.zero_grad()
                self.mask_optimizer.zero_grad()
                outer_loss.backward()
                self.restoration_optimizer.step()
                self.mask_optimizer.step()
                self._store_state()
                ### Outer Loop End

        tqdm_batch.close()

    @torch.no_grad()
    def validate(self, val_loader):
        # Todo
        pass

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