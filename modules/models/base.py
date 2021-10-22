import torch

from torch import nn

class BaseModel(nn.Module):
    def __init__(self, args, device):
        super(BaseModel, self).__init__()
        self.args = args
        self.device = device
        
    def load_checkpoint(self, file_path):
        """
        Load checkpoint
        """
        raise NotImplementedError
    
    def save_checkpoint(self, file_name, is_best=0):
        """
        Save checkpoint
        """
        raise NotImplementedError

    def adjust_learning_rate(self, epoch):
        """
        Adjust learning rate every epoch
        """
        raise NotImplementedError

    def train_one_epoch(self, train_loader, epoch):
        """
        Training step for each mini-batch
        """
        raise NotImplementedError

    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validation step for each mini-batch
        """
        raise NotImplementedError

    def init_training_logger(self):
        """
        Initialize training logger specific for each model
        """
        raise NotImplementedError

    def init_validation_logger(self):
        """
        Initialize validation logger specific for each model
        """
        raise NotImplementedError

    def init_testing_logger(self):
        """
        Initialize testing logger specific for each model
        """
        raise NotImplementedError

    def finalize_training(self):
        """
        Finalize training
        """
        raise NotImplementedError

    def move_components_to_device(self, mode):
        """
        Move components to device
        """
        raise NotImplementedError

    def _reset_metric(self):
        """
        Metric related to average meter
        """
        raise NotImplementedError

    def count_parameters(self):
        """
        Return the number of parameters for the model
        """
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)