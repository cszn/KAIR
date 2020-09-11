import os
import torch
import torch.nn as nn
from utils.utils_bnorm import merge_bn, tidy_sequential


class ModelBase():
    def __init__(self, opt):
        self.opt = opt                         # opt
        self.save_dir = opt['path']['models']  # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']        # training or not
        self.schedulers = []                   # schedulers

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def init_train(self):
        pass

    def load(self):
        pass

    def save(self, label):
        pass

    def define_loss(self):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def current_visuals(self):
        pass

    def current_losses(self):
        pass

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]


    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    def describe_network(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        for name, param in network.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        return msg

    """
    # ----------------------------------------
    # Save prameters
    # Load prameters
    # ----------------------------------------
    """

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)

    """
    # ----------------------------------------
    # Merge Batch Normalization for training
    # Merge Batch Normalization for testing
    # ----------------------------------------
    """

    # ----------------------------------------
    # merge bn during training
    # ----------------------------------------
    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    # ----------------------------------------
    # merge bn before testing
    # ----------------------------------------
    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
