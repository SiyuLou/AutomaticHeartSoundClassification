import os
import copy
import shutil
import argparse
import collections
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device,ensure_dir 
from model.model import reset_parameters
from test import evaluation

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')
     
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
   
    # split the data
    label_df = pd.read_csv(config['data_loader']['full_label_csv'])
    label_df = shuffle(label_df)
    
    for k, (train_idx, test_idx) in enumerate(kfold.split(label_df, label_df['label'])):
        logger.info("Now start %d th validation"%k)
        # build model architecture, then print to console
        model = config.init_obj('arch', module_arch)
        logger.info(model)

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
 
        # setup data_loader instances
        train = label_df.iloc[train_idx]
        test = label_df.iloc[test_idx]
        data_loader = config.init_obj(name = 'data_loader', 
                                      module = module_data,
                                      label_df = train)
        valid_data_loader = data_loader.split_validation()
        # initialize model
        reset_parameters(model)
        
        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=config,
                          device=device,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler)

        trainer.train()
        
        # create folder for each folder
        dir_path = os.path.join(config.save_dir,'fold_%d'%k) 
        ensure_dir(dir_path)
        file_path = os.path.join(dir_path, 'model_best.pth')
        best_ckpt_path = config.save_dir / "model_best.pth"
        try:
            shutil.copyfile(best_ckpt_path, file_path)
            print("File copied successfully.")
        except shutil.SameFileError:
            print("Souce and destination represents the same file.")
        except IsADirectoryError:
            print("Destination is a directory.")
        
        logger = config.get_logger('test')
        data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['fea_path'],
            label_df = test,
            duration=config['data_loader']['args']['duration'],
            batch_size=32,
            delta=config['data_loader']['args']['delta'],
            norm= config['data_loader']['args']['norm'],
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2
        )

        model = config.init_obj('arch', module_arch)

        evaluation(data_loader, model, criterion, metrics,best_ckpt_path, logger)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
