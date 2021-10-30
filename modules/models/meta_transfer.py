import logging
import os
import shutil
import time
import dataloaders
import torch


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

    def train_one_epoch(self, train_loaders, epoch):
        """
        Training step for each mini-batch
        """
        self.current_epoch = epoch
        self._reset_metric()

        tqdm_batch_d1 = tqdm(
            train_loaders[0],
            desc="epoch-{} conventional training".format(self.current_epoch),
        )

        tqdm_batch_d2 = tqdm(
            train_loaders[1],
            desc="epoch-{} conventional training".format(self.current_epoch),
        )

        self.restoration_net.train()
        self.mask_net.train()

        end_time = time.time()

        self.restoration_optimizer.zero_grad()
        self.mask_optimizer.zero_grad()
        # Freeze network body
        for name, param in self.restoration_net.name_parameters()[:-2]:
            param.require_grad = False

        # Run inner loop
        # for curr_dt, tqdm_batch in enumerate([tqdm_batch_d1, tqdm_batch_d2]):
        for curr_dt, tqdm_batch in enumerate([tqdm_batch_d1, tqdm_batch_d2]):
            # for curr_it, data in enumerate(tqdm_batch):
            data = next(iter(tqdm_batch))
            self.data_time_meter.update(time.time() - end_time)
            print(type(data))
            noisy = data["noisy"]
            clean = data["clean"]

            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # Predict clean and noisy image
            out_clean, noisy_rec = self.restoration_net(noisy)

            # Generate mask
            mask = self.mask_net(noisy)
            mask = (mask > 0.5).float()
            num_non_zero = torch.count_nonzero(mask)
            num_zero = mask.size()[0] * 256 * 256 - num_non_zero

            # Compute auxiliary loss
            noisy_rec_loss = torch.sum(
                (torch.abs((noisy_rec - noisy) * mask)) / torch.sum(mask)
            )

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

            loss = noisy_rec_loss

        # End of inner loop

        # Unfreeze all network
        for name, param in self.restoration_net.name_parameters():
            param.require_grad = True

        # iterate over sample
        for curr_dt, tqdm_batch in enumerate([tqdm_batch_d1, tqdm_batch_d2]):
            # for curr_it, data in enumerate(tqdm_batch):
            data = next(iter(tqdm_batch))

            self.data_time_meter.update(time.time() - end_time)

            noisy = data["noisy"]
            clean = data["clean"]

            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.restoration_optimizer.zero_grad()
            self.mask_optimizer.zero_grad()

            # Predict clean and noisy image
            out_clean, noisy_rec = self.restoration_net(noisy)

            # compute primary loss
            out_clean, noisy_rec = self.restoration_net(noisy)
            clean_loss = self.rec_criterion(out_clean, clean)
            clean_loss.backward()  # add entropy loss here if want

        self.restoration_optimizer.step()
        self.mask_optimizer.step()

        # End of outer loop

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