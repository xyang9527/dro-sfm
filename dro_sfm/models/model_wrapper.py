
from collections import OrderedDict
import os
import time
import random
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
import logging

from dro_sfm.datasets.transforms import get_transforms
from dro_sfm.utils.depth import inv2depth, post_process_inv_depth, compute_depth_metrics, compute_pose_metrics, compute_depth_metrics_demon
from dro_sfm.utils.horovod import print0, world_size, rank, on_rank_0
from dro_sfm.utils.image import flip_lr, flip_lr_intr
from dro_sfm.utils.load import load_class, load_class_args_create, \
    load_network, filter_args
from dro_sfm.utils.logging import pcolor
from dro_sfm.utils.reduce import all_reduce_metrics, reduce_dict, \
    create_dict, average_loss_and_metrics
from dro_sfm.utils.save import save_depth
from dro_sfm.models.model_utils import stack_batch
import IPython


class ModelWrapper(torch.nn.Module):
    """
    Top-level torch.nn.Module wrapper around a SfmModel (pose+depth networks).
    Designed to use models with high-level Trainer classes (cf. trainers/).

    Parameters
    ----------
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    """

    def __init__(self, config, resume=None, logger=None, load_datasets=True):
        logging.warning(f'ModelWrapper::__init__(.., load_datasets={load_datasets})')
        super().__init__()

        # Store configuration, checkpoint and logger
        self.config = config
        self.logger = logger
        self.resume = resume

        # Set random seed
        set_random_seed(config.arch.seed)

        # Task metrics
        self.metrics_name = 'depth'
        self.metrics_keys = ('abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3', 'SILog', 'l1_inv', 'rot_ang', 't_ang', 't_cm')
        self.metrics_modes = ('', '_pp', '_gt', '_pp_gt')

        # Model, optimizers, schedulers and datasets are None for now
        self.model = self.optimizer = self.scheduler = None
        self.train_dataset = self.validation_dataset = self.test_dataset = None
        self.current_epoch = 0

        # logging
        logging.info(f'  self.config:              {type(self.config)}')
        logging.info(f'  self.logger:              {self.logger}')
        logging.info(f'  self.resume:              {self.resume}')
        logging.info(f'  self.metrics_name:        {self.metrics_name}')
        logging.info(f'  self.metrics_keys:        {self.metrics_keys}')
        logging.info(f'  self.metrics_modes:       {self.metrics_modes}')

        # Prepare model
        self.prepare_model(resume)

        # Prepare datasets
        if load_datasets:
            # Requirements for validation (we only evaluate depth for now)
            validation_requirements = {'gt_depth': True, 'gt_pose': False}
            test_requirements = validation_requirements
            self.prepare_datasets(validation_requirements, test_requirements)

        # Preparations done
        self.config.prepared = True

        logging.info(f'  self.current_epoch:       {self.current_epoch}')

        logging.info(f'  self.train_dataset:       {self.train_dataset}')
        logging.info(f'  self.validation_dataset:  {self.validation_dataset}')
        logging.info(f'  self.test_dataset:        {self.test_dataset}')

        logging.info(f'  self.model:               {type(self.model)}')
        logging.info(f'  self.optimizer:           {self.optimizer}')
        logging.info(f'  self.scheduler:           {self.scheduler}')
        logging.info(f'  self.config:              {type(self.config)}')


    def prepare_model(self, resume=None):
        """Prepare self.model (incl. loading previous state)"""
        logging.warning(f'prepare_model(resume={resume})')
        print0(pcolor('### Preparing Model', 'green'))
        self.model = setup_model(self.config.model, self.config.prepared)
        # Resume model if available
        if resume:
            print0(pcolor('### Resuming from {}'.format(
                resume['file']), 'magenta', attrs=['bold']))
            self.model = load_network(
                self.model, resume['state_dict'], 'model')
            if 'epoch' in resume:
                self.current_epoch = resume['epoch']

    def prepare_datasets(self, validation_requirements, test_requirements):
        """Prepare datasets for training, validation and test."""
        logging.warning(f'prepare_datasets(validation_requirements={validation_requirements}, test_requirements={test_requirements})')
        # Prepare datasets
        print0(pcolor('### Preparing Datasets', 'green'))

        augmentation = self.config.datasets.augmentation
        # Setup train dataset (requirements are given by the model itself)
        self.train_dataset = setup_dataset(
            self.config.datasets.train, 'train',
            self.model.train_requirements, **augmentation)
        # Setup validation dataset
        self.validation_dataset = setup_dataset(
            self.config.datasets.validation, 'validation',
            validation_requirements, **augmentation)
        # Setup test dataset
        self.test_dataset = setup_dataset(
            self.config.datasets.test, 'test',
            test_requirements, **augmentation)

    @property
    def depth_net(self):
        """Returns depth network."""
        logging.warning(f'depth_net()')
        return self.model.depth_net

    @property
    def pose_net(self):
        """Returns pose network."""
        logging.warning(f'pose_net()')
        return self.model.pose_net

    @property
    def percep_net(self):
        """Returns perceptual network."""
        logging.warning(f'percept_net()')
        return self.model.percep_net
    
    @property
    def logs(self):
        """Returns various logs for tracking."""
        params = OrderedDict()
        for param in self.optimizer.param_groups:
            params['{}_learning_rate'.format(param['name'].lower())] = param['lr']
        params['progress'] = self.progress
        return {
            **params,
            **self.model.logs,
        }

    @property
    def progress(self):
        """Returns training progress (current epoch / max. number of epochs)"""
        return self.current_epoch / self.config.arch.max_epochs

    def configure_optimizers(self):
        """Configure depth and pose optimizers and the corresponding scheduler."""
        logging.warning(f'configure_optimizers()')

        params = []
        # Load optimizer
        optimizer = getattr(torch.optim, self.config.model.optimizer.name)

        # Depth optimizer
        if self.depth_net is not None:
            logging.info(f'  add depth optimizer')
            params.append({
                'name': 'Depth',
                'params': self.depth_net.parameters(),
                **filter_args(optimizer, self.config.model.optimizer.depth)
            })

        # Pose optimizer
        if self.pose_net is not None:
            logging.info(f'  add pose optimizer')
            params.append({
                'name': 'Pose',
                'params': [param for param in self.pose_net.parameters() if param.requires_grad],
                **filter_args(optimizer, self.config.model.optimizer.pose)
            })
        else:
            logging.warning(f'  skip pose optimizer')

        # Create optimizer with parameters
        optimizer = optimizer(params)

        # Load and initialize scheduler
        scheduler = getattr(torch.optim.lr_scheduler, self.config.model.scheduler.name)
        scheduler = scheduler(optimizer, **filter_args(scheduler, self.config.model.scheduler))

        # if self.resume:
        #     try:
        #         if 'optimizer' in self.resume:
        #             optimizer.load_state_dict(self.resume['optimizer'])
        #         if 'scheduler' in self.resume:
        #             scheduler.load_state_dict(self.resume['scheduler'])
        #     except Exception as e:
        #         print(e)

        # Create class variables so we can use it internally
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Return optimizer and scheduler
        return optimizer, scheduler

    def train_dataloader(self):
        """Prepare training dataloader."""
        logging.warning(f'train_dataloader()')
        return setup_dataloader(self.train_dataset,
                                self.config.datasets.train, 'train')[0]

    def val_dataloader(self):
        """Prepare validation dataloader."""
        logging.warning(f'val_dataloader()')
        return setup_dataloader(self.validation_dataset,
                                self.config.datasets.validation, 'validation')

    def test_dataloader(self):
        """Prepare test dataloader."""
        logging.warning(f'test_dataloader()')
        return setup_dataloader(self.test_dataset,
                                self.config.datasets.test, 'test')

    def training_step(self, batch, *args):
        """Processes a training batch."""
        batch = stack_batch(batch)
        output = self.model(batch, progress=self.progress)
        if output is None:
            return None
        return {
            'loss': output['loss'],
            'metrics': output['metrics']
        }

    def validation_step(self, batch, *args):
        """Processes a validation batch."""
        output = self.evaluate_depth(batch)
        if self.logger:
            self.logger.log_depth('val', batch, output, args,
                                  self.validation_dataset, world_size(),
                                  self.config.datasets.validation)
        return {
            'idx': batch['idx'],
            **output['metrics'],
        }

    def test_step(self, batch, *args):
        """Processes a test batch."""
        output = self.evaluate_depth(batch)
        save_depth(batch, output, args,
                   self.config.datasets.test,
                   self.config.save)
        return {
            'idx': batch['idx'],
            **output['metrics'],
        }

    def training_epoch_end(self, output_batch):
        """Finishes a training epoch."""
        logging.warning(f'training_epoch_end(..)')
        # Calculate and reduce average loss and metrics per GPU
        loss_and_metrics = average_loss_and_metrics(output_batch, 'avg_train')
        loss_and_metrics = reduce_dict(loss_and_metrics, to_item=True)

        # Log to wandb
        if self.logger:
            self.logger.log_metrics({
                **self.logs, **loss_and_metrics,
            })

        return {
            **loss_and_metrics
        }

    def validation_epoch_end(self, output_data_batch):
        """Finishes a validation epoch."""
        logging.warning(f'validation_epoch_end({len(output_data_batch)})')

        # Reduce depth metrics
        metrics_data = all_reduce_metrics(
            output_data_batch, self.validation_dataset, self.metrics_name)

        # Create depth dictionary
        metrics_dict = create_dict(
            metrics_data, self.metrics_keys, self.metrics_modes,
            self.config.datasets.validation)

        # Print stuff
        self.print_metrics(metrics_data, self.config.datasets.validation)

        # Log to wandb
        if self.logger:
            self.logger.log_metrics({
                **metrics_dict, 'global_step': self.current_epoch + 1,
            })

        return {
            **metrics_dict
        }

    def test_epoch_end(self, output_data_batch):
        """Finishes a test epoch."""
        logging.warning(f'test_epoch_end({len(output_data_batch)})')

        # Reduce depth metrics
        metrics_data = all_reduce_metrics(
            output_data_batch, self.test_dataset, self.metrics_name)

        # Create depth dictionary
        metrics_dict = create_dict(
            metrics_data, self.metrics_keys, self.metrics_modes,
            self.config.datasets.test)

        # Print stuff
        self.print_metrics(metrics_data, self.config.datasets.test)

        return {
            **metrics_dict
        }

    def forward(self, *args, **kwargs):
        """Runs the model and returns the output."""
        assert self.model is not None, 'Model not defined'
        return self.model(*args, **kwargs)

    def depth(self, *args, **kwargs):
        """Runs the pose network and returns the output."""
        assert self.depth_net is not None, 'Depth network not defined'
        logging.info(f'  depth_net: {type(self.depth_net)}')
        logging.info(f'  args:      {args}')
        logging.info(f'  kwargs:    {kwargs}')
        return self.depth_net(*args, **kwargs)

    def pose(self, *args, **kwargs):
        """Runs the depth network and returns the output."""
        assert self.pose_net is not None, 'Pose network not defined'
        logging.info(f'  pose_net:  {type(self.pose_net)}')
        logging.info(f'  args:      {args}')
        logging.info(f'  kwargs:    {kwargs}')
        return self.pose_net(*args, **kwargs)

    def percep(self, *args, **kwargs):
        """Runs the depth network and returns the output."""
        assert self.percep_net is not None, 'Perceptual network not defined'
        logging.info(f'  percep_net:  {type(self.percep_net)}')
        logging.info(f'  args:        {args}')
        logging.info(f'  kwargs:      {kwargs}')
        return self.percep_net(*args, **kwargs)
    
    def evaluate_depth(self, batch):
        """Evaluate batch to produce depth metrics."""
        # Get predicted depth
        output = self.model(batch)
        inv_depths = output['inv_depths']
        poses = output['poses']
        depth = inv2depth(inv_depths[0])

        # Post-process predicted depth
        batch['rgb'] = flip_lr(batch['rgb'])
        if 'rgb_context' in batch:
            batch['rgb_context'] = [flip_lr(img) for img in batch['rgb_context']]
        batch['intrinsics'] = flip_lr_intr(batch['intrinsics'], width=depth.shape[3])
        inv_depths_flipped = self.model(batch)['inv_depths']
        inv_depth_pp = post_process_inv_depth(
            inv_depths[0], inv_depths_flipped[0], method='mean')
        depth_pp = inv2depth(inv_depth_pp)
        batch['rgb'] = flip_lr(batch['rgb'])

        # Calculate predicted metrics
        if 'pose_context' in batch.keys():
            pose_errs = compute_pose_metrics(self.config.model.params, gt=batch['pose_context'], pred=poses)
        else:
            pose_errs = [0, 0, 0]
        metrics = OrderedDict()

        if 'depth' in batch:
            for mode in self.metrics_modes:
                if self.config['datasets']['validation']['dataset'] == ['Demon']:
                    metrics[self.metrics_name + mode] = compute_depth_metrics_demon(
                    self.config.model.params, gt=batch['depth'], gt_pose=batch['pose_context'],
                    pred=depth_pp if 'pp' in mode else depth,
                    use_gt_scale='gt' in mode)
                else:
                    metrics[self.metrics_name + mode] = compute_depth_metrics(
                        self.config.model.params, gt=batch['depth'],
                        pred=depth_pp if 'pp' in mode else depth,
                        use_gt_scale='gt' in mode)
                metrics[self.metrics_name + mode] = torch.cat([metrics[self.metrics_name + mode], 
                                                        torch.Tensor(pose_errs).to(depth_pp.device)])
        # Return metrics and extra information
        return {
            'metrics': metrics,
            'inv_depth': inv_depth_pp
        }

    @on_rank_0
    def print_metrics(self, metrics_data, dataset):
        """Print depth metrics on rank 0 if available"""
        if not metrics_data[0]:
            return

        hor_line = '|{:<}|'.format('*' * 148)
        met_line = '| {:^14} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} |'
        num_line = '{:<14} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f}'

        def wrap(string):
            return '| {} |'.format(string)

        print()
        print()
        print()
        print(hor_line)

        if self.optimizer is not None:
            bs = 'E: {} BS: {}'.format(self.current_epoch + 1,
                                       self.config.datasets.train.batch_size)
            if self.model is not None:
                bs += ' - {}'.format(self.config.model.name)
            lr = 'LR ({}):'.format(self.config.model.optimizer.name)
            for param in self.optimizer.param_groups:
                lr += ' {} {:.2e}'.format(param['name'], param['lr'])
            par_line = wrap(pcolor('{:<40}{:>51}'.format(bs, lr),
                                   'green', attrs=['bold', 'dark']))
            print(par_line)
            print(hor_line)

        print(met_line.format(*(('METRIC',) + self.metrics_keys)))
        for n, metrics in enumerate(metrics_data):
            print(hor_line)
            path_line = '{}'.format(
                os.path.join(dataset.path[n], dataset.split[n]))
            if len(dataset.cameras[n]) == 1: # only allows single cameras
                path_line += ' ({})'.format(dataset.cameras[n][0])
            print(wrap(pcolor('*** {:<87}'.format(path_line), 'magenta', attrs=['bold'])))
            print(hor_line)
            for key, metric in metrics.items():
                if self.metrics_name in key:
                    print(wrap(pcolor(num_line.format(
                        *((key.upper(),) + tuple(metric.tolist()))), 'cyan')))
        print(hor_line)

        if self.logger:
            run_line = wrap(pcolor('{:<60}{:>31}'.format(
                self.config.wandb.url, self.config.wandb.name), 'yellow', attrs=['dark']))
            print(run_line)
            print(hor_line)

        print()


def set_random_seed(seed):
    if seed >= 0:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_depth_net(config, prepared, **kwargs):
    """
    Create a depth network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    prepared : bool
        True if the network has been prepared before
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    depth_net : nn.Module
        Create depth network
    """
    logging.warning(f'setup_depth_net( {config.name}, prepared={prepared}, {config.checkpoint_path}, .. )')
    print0(pcolor('DepthNet: %s' % config.name, 'yellow'))
    depth_net = load_class_args_create(config.name,
        paths=['dro_sfm.networks.depth_pose',],
        args={**config, **kwargs},
    )
    if not prepared and config.checkpoint_path is not '':
        depth_net = load_network(depth_net, config.checkpoint_path,
                                 ['depth_net', 'disp_network'])
    return depth_net


def setup_pose_net(config, prepared, **kwargs):
    """
    Create a pose network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    prepared : bool
        True if the network has been prepared before
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    pose_net : nn.Module
        Created pose network
    """
    logging.warning(f'setup_pose_net({config.name}, prepared={prepared}, {config.checkpoint_path}, ..)')
    print0(pcolor('PoseNet: %s' % config.name, 'yellow'))
    pose_net = load_class_args_create(config.name,
        paths=['dro_sfm.networks.pose',],
        args={**config, **kwargs},
    )
    if not prepared and config.checkpoint_path is not '':
        pose_net = load_network(pose_net, config.checkpoint_path,
                                ['pose_net', 'pose_network'])
    return pose_net


def setup_percep_net(config, prepared, **kwargs):
    """
    Create a perceputal network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    prepared : bool
        True if the network has been prepared before
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    depth_net : nn.Module
        Create depth network
    """
    logging.warning(f'setup_percep_net({config.name}, prepared={prepared}, ..)')
    print0(pcolor('PercepNet: %s' % config.name, 'yellow'))
    percep_net = load_class_args_create(config.name,
        paths=['dro_sfm.networks.layers',],
        args={**config, **kwargs},
    )
    return percep_net

def setup_model(config, prepared, **kwargs):
    """
    Create a model

    Parameters
    ----------
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    prepared : bool
        True if the model has been prepared before
    kwargs : dict
        Extra parameters for the model

    Returns
    -------
    model : nn.Module
        Created model
    """
    logging.warning(f'setup_model({config.name}, prepared={prepared}, ..)')
    logging.info(f'  config:\n{config}')
    logging.info(f'  kwargs:\n{kwargs}')
    print0(pcolor('Model: %s' % config.name, 'yellow'))
    config.loss.min_depth = config.params.min_depth
    config.loss.max_depth = config.params.max_depth
    model = load_class(config.name, paths=['dro_sfm.models',])(
        **{**config.loss, **kwargs})

    logging.info(f'  model.network_requirements: {model.network_requirements}')
    logging.info(f'  model.train_requirements:   {model.train_requirements}')

    # Add depth network if required
    if model.network_requirements['depth_net']:
        config.depth_net.max_depth = config.params.max_depth
        config.depth_net.min_depth = config.params.min_depth
        model.add_depth_net(setup_depth_net(config.depth_net, prepared))

    # Add pose network if required
    if model.network_requirements['pose_net']:
        model.add_pose_net(setup_pose_net(config.pose_net, prepared))
    # Add percep_net if required

    if model.network_requirements['percep_net']:
        model.add_percep_net(setup_percep_net(config.percep_net, prepared))

    # If a checkpoint is provided, load pretrained model
    if not prepared and config.checkpoint_path is not '':
        model = load_network(model, config.checkpoint_path, 'model')
    # Return model
    return model


def setup_dataset(config, mode, requirements, **kwargs):
    """
    Create a dataset class

    Parameters
    ----------
    config : CfgNode
        Configuration (cf. configs/default_config.py)
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the dataset
    requirements : dict (string -> bool)
        Different requirements for dataset loading (gt_depth, gt_pose, etc)
    kwargs : dict
        Extra parameters for dataset creation

    Returns
    -------
    dataset : Dataset
        Dataset class for that mode
    """
    logging.warning(f'setup_dataset({config.path}, {config.split}, mode={mode}, requirements={requirements}, ..)')
    # If no dataset is given, return None
    if len(config.path) == 0:
        return None

    print0(pcolor('###### Setup %s datasets' % mode, 'red'))

    # Global shared dataset arguments
    dataset_args = {
        'back_context': config.back_context,
        'forward_context': config.forward_context,
        'data_transform': get_transforms(mode, **kwargs),
        'strides': config.strides
    }

    # Loop over all datasets
    datasets = []
    for i in range(len(config.split)): 
        path_split = os.path.join(config.path[i], config.split[i])

        # Individual shared dataset arguments
        dataset_args_i = {
            'depth_type': config.depth_type[i] if requirements['gt_depth'] else None,
            'with_pose': requirements['gt_pose'],
        }
        logging.info(f'config.dataset[i]:    {config.dataset[i]}')

        # KITTI dataset
        if config.dataset[i] == 'KITTI':
            logging.info(f'  load dataset KITTI')
            from dro_sfm.datasets.kitti_dataset import KITTIDataset
            dataset = KITTIDataset(
                config.path[i], path_split,
                **dataset_args, **dataset_args_i,
            )
        # DGP dataset
        elif config.dataset[i] == 'DGP':
            logging.info(f'  load dataset DGP')
            from dro_sfm.datasets.dgp_dataset import DGPDataset
            dataset = DGPDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
                cameras=config.cameras[i],
            )
        # NYU dataset
        elif config.dataset[i] == 'NYU':
            logging.info(f'  load dataset NYU')
            from dro_sfm.datasets.nyu_dataset_processed import NYUDataset
            dataset = NYUDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )
        # NYU dataset
        elif config.dataset[i] == 'NYUtest':
            logging.info(f'  load dataset NYUtest')
            from dro_sfm.datasets.nyu_dataset_test_processed import NYUDataset
            dataset = NYUDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )
        # Demon dataset
        elif config.dataset[i] == 'Demon':
            logging.info(f'  load dataset Demon')
            from dro_sfm.datasets.demon_dataset import DemonDataset
            dataset = DemonDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )
        
        # DemonMF dataset
        elif config.dataset[i] == 'DemonMF':
            logging.info(f'  load dataset DemonMF')
            from dro_sfm.datasets.demon_mf_dataset import DemonDataset
            dataset = DemonDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )


        # Scannet dataset
        elif config.dataset[i] == 'Scannet':
            logging.info(f'  load dataset Scannet')
            from dro_sfm.datasets.scannet_dataset import ScannetDataset
            dataset = ScannetDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )
        # Scannet dataset
        elif config.dataset[i] == 'ScannetTest':
            logging.info(f'  load dataset ScannetTest')
            from dro_sfm.datasets.scannet_test_dataset import ScannetTestDataset
            dataset = ScannetTestDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )

        # Scannet dataset
        elif config.dataset[i] == 'ScannetTestMF':
            logging.info(f'  load dataset ScannetTestMF')
            from dro_sfm.datasets.scannet_test_dataset_mf import ScannetTestDataset
            dataset = ScannetTestDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )
            
        # Scannet banet dataset - BA-Net: Dense Bundle Adjustment Networks
        elif config.dataset[i] == 'ScannetBA':
            logging.info(f'  load dataset ScannetBA')
            from dro_sfm.datasets.scannet_banet_dataset import ScannetBADataset
            dataset = ScannetBADataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )

        # ======================================================================
        # Matterport dataset
        elif config.dataset[i] == 'Matterport':
            logging.info(f'  load dataset Matterport')
            from dro_sfm.datasets.matterport_dataset import MatterportDataset
            dataset = MatterportDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )
        # Matterport dataset
        elif config.dataset[i] == 'MatterportTest':
            logging.info(f'  load dataset MatterportTest')
            from dro_sfm.datasets.matterport_test_dataset import MatterportTestDataset
            dataset = MatterportTestDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )
            
        # Matterport banet dataset
        elif config.dataset[i] == 'MatterportBA':
            logging.info(f'  load dataset MatterportBA')
            from dro_sfm.datasets.matterport_banet_dataset import MatterportBADataset
            dataset = MatterportBADataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )
        # ======================================================================

        # Video dataset
        elif config.dataset[i] == 'Video':
            logging.info(f'  load dataset Video')
            from dro_sfm.datasets.video_dataset import VideoDataset
            dataset = VideoDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )

        # Video random sample dataset
        elif config.dataset[i] == 'Video_Random':
            logging.info(f'  load dataset Video_Random')
            from dro_sfm.datasets.video_random_dataset import VideoRandomDataset
            dataset = VideoRandomDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )
                         
        # Image dataset
        elif config.dataset[i] == 'Image':
            logging.info(f'  load dataset Image')
            from dro_sfm.datasets.image_dataset import ImageDataset
            dataset = ImageDataset(
                config.path[i], config.split[i],
                **dataset_args, **dataset_args_i,
            )
        else:
            ValueError('Unknown dataset %d' % config.dataset[i])

        # Repeat if needed
        if 'repeat' in config and config.repeat[i] > 1:
            dataset = ConcatDataset([dataset for _ in range(config.repeat[i])])
        datasets.append(dataset)

        # Display dataset information
        bar = '######### {:>7}'.format(len(dataset))
        if 'repeat' in config:
            bar += ' (x{})'.format(config.repeat[i])
        bar += ': {:<}'.format(path_split)
        print0(pcolor(bar, 'yellow'))

    # If training, concatenate all datasets into a single one
    if mode == 'train':
        datasets = [ConcatDataset(datasets)]

    return datasets


def worker_init_fn(worker_id):
    """Function to initialize workers"""
    # logging.warning(f'worker_init_fn({worker_id})')
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)


def get_datasampler(dataset, mode):
    """Distributed data sampler"""
    return torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=(mode=='train'),
        num_replicas=world_size(), rank=rank())


def setup_dataloader(datasets, config, mode):
    """
    Create a dataloader class

    Parameters
    ----------
    datasets : list of Dataset
        List of datasets from which to create dataloaders
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the dataloader

    Returns
    -------
    dataloaders : list of Dataloader
        List of created dataloaders for each input dataset
    """
    logging.warning(f'setup_dataloader(\n  datasets={datasets},\n  config.batch_size={config.batch_size},\n  config.num_works={config.num_workers}, {mode})')
    return [(DataLoader(dataset,
                        batch_size=config.batch_size, shuffle=False,
                        pin_memory=True, num_workers=config.num_workers,
                        worker_init_fn=worker_init_fn,
                        sampler=get_datasampler(dataset, mode))
             ) for dataset in datasets]
