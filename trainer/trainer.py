import torch
import logging
import os
import random

from utils.misc import print_cuda_statistics


class Trainer(object):
    """
    Wrapper for training, more related to engineering than research code
    """

    def __init__(self, model, train_loader, val_loader, args):
        self.args = args
        self.logger = logging.getLogger("Trainer")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model

        self.is_train = False
        self.start_epoch = 0
        self.end_epoch = args.max_epoch

        self.logger = logging.getLogger("Trainer")
        if hasattr(args, "resume"):
            self.logger.info("Resuming from checkpoint")
            assert os.path.isfile(args.resume)
            self.start_epoch = self.model.load_checkpoint(args.resume)
        self.logger.info(str(self.args))
        self.logger.info(
            "Number of model parameters: {}".format(self.model.count_parameters())
        )

    def train(self):
        self.is_train = True
        self.model.init_training_logger()
        self.best_performance = 0
        for epoch in range(self.start_epoch, self.end_epoch):
            if self.args.meta_transfer == 1:
                self.model.train_one_epoch(self.train_loader, epoch)
            else:
                self.model.train_one_epoch(self.train_loader, epoch)
            self.model.adjust_learning_rate(epoch)

            continue

            if ((epoch + 1) % self.args.validate_every) == 0:
                performance = self.validate()
                if performance > self.best_performance:
                    self.best_performance = performance
                    self.model.save_checkpoint("model_{}.pth".format(epoch), is_best=1)
                    self.logger.info(
                        "best performance achieved at epoch {} with performance of {}".format(
                            epoch, self.best_performance
                        )
                    )

            if ((epoch + 1) % self.args.save_every) == 0:
                self.model.save_checkpoint("model_{}.pth".format(epoch))

        self.finalize_training()

    def validate(self):
        # return performance to save the best model, if there is no performance measure e.g. GAN then just return 0
        if not self.is_train:  # if mode == validation only
            self.model.init_validation_logger()
        return self.model.validate(self.val_loader)

    def test(self):
        self.model.init_testing_logger()
        self.model.validate(self.val_loader)

    def finalize_training(self):
        self.model.finalize_training()

    # TODO: add sanity check (run testbed)