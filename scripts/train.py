
import sys
import os
lib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_dir)
import argparse
from dro_sfm.models.model_wrapper import ModelWrapper
from dro_sfm.models.model_checkpoint import ModelCheckpoint
from dro_sfm.trainers.horovod_trainer import HorovodTrainer
from dro_sfm.utils.config import parse_train_file
from dro_sfm.utils.load import set_debug, filter_args_create
from dro_sfm.utils.horovod import hvd_init, rank
from dro_sfm.loggers import WandbLogger

import time
import logging
import datetime
import os.path as osp


def setup_log():
    # Initialize logging
    # simple_format = '%(levelname)s >>> %(message)s'
    medium_format = (
        '%(levelname)s : %(filename)s[%(lineno)d]'
        ' >>> %(message)s'
    )

    # Reference:
    #   http://59.125.118.185:8088/ALG/TestingTools/-/blob/master/model_performance_evaluation_tool/src/common/testingtools_log.py
    formatter = logging.Formatter(
                '[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')

    medium_format_new = (
        '[%(asctime)s] %(levelname)s : %(filename)s[%(lineno)d] %(funcName)s'
        ' >>> %(message)s'
    )

    get_log_file = osp.join(osp.dirname(__file__), '../train_kneron.log')

    logging.basicConfig(
        filename=get_log_file,
        filemode='w',
        level=logging.INFO,
        format=medium_format_new
    )
    logging.info('@{} created at {}'.format(
        get_log_file,
        datetime.datetime.now())
    )
    print('\n===== log_file: {}\n'.format(get_log_file))


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='dro-sfm training script')
    parser.add_argument('file', type=str, help='Input file (.ckpt or .yaml)')
    parser.add_argument('--config', type=str, default=None, help='Input file (yaml)')
    args = parser.parse_args()
    assert args.file.endswith(('.ckpt', '.yaml')), \
        'You need to provide a .ckpt of .yaml file'
    return args


def train(file, config):
    """
    Monocular depth estimation training script.

    Parameters
    ----------
    file : str
        Filepath, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file.
    """
    logging.warning(f'train({file}, {type(config)})')
    # Initialize horovod
    hvd_init()

    # Produce configuration and checkpoint from filename
    config, ckpt = parse_train_file(file, config)

    logging.info(f'config.debug: {config.debug}')

    # Set debug if requested
    set_debug(config.debug)

    # Wandb Logger
    logger = None if config.wandb.dry_run or rank() > 0 \
        else filter_args_create(WandbLogger, config.wandb)

    # model checkpoint
    checkpoint = None if config.checkpoint.filepath is '' or rank() > 0 else \
        filter_args_create(ModelCheckpoint, config.checkpoint)

    # Initialize model wrapper
    model_wrapper = ModelWrapper(config, resume=ckpt, logger=logger)

    # Create trainer with args.arch parameters
    trainer = HorovodTrainer(**config.arch, checkpoint=checkpoint)

    # Train model
    trainer.fit(model_wrapper)


if __name__ == '__main__':
    setup_log()
    args = parse_args()
    time_beg_main = time.time()
    train(args.file, args.config)
    time_end_main = time.time()
    logging.warning(f'elapsed {time_end_main - time_beg_main:.3f} seconds.')
