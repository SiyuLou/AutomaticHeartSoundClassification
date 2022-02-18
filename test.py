import pandas as pd
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


def test(config, resume=None, df=None):
    logger = config.get_logger('test')
    if df is None:
        label_df = pd.read_csv('data/test.csv')
    else:
        label_df = df
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['fea_path'],
        label_df = label_df,
        duration=config['data_loader']['args']['duration'],
        batch_size=32,
        delta=config['data_loader']['args']['delta'],
        norm= config['data_loader']['args']['norm'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    if resume is None:
        checkpoint = torch.load(config.resume)
    else:
        checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    val_pred_pos = 0
    val_pred_neg = 0
    val_condition_pos = 0
    val_condition_neg = 0
    val_TP = 0
    val_TN = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            
            _,indices = torch.max(output, 1)
            
            pred_pos_idx = [i for i, x in enumerate(indices) if x == 0]
            condition_pos_idx = [i for i, x in enumerate(target) if x == 0]
            pred_neg_idx = [i for i, x in enumerate(indices) if x == 1]
            condition_neg_idx = [i for i, x in enumerate(target) if x == 1]
            
            val_pred_pos += len(pred_pos_idx)
            val_condition_pos += len(condition_pos_idx)
            val_pred_neg += len(pred_neg_idx)
            val_condition_neg += len(condition_neg_idx)
            val_TP += len([x for x in pred_pos_idx if x in condition_pos_idx])
            val_TN += len([x for x in pred_neg_idx if x in condition_neg_idx])
            correct = (indices == target).float()
            val_correct += torch.sum(correct).item()
            val_total += target.shape[0]
            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    sensitivity = val_TP / val_condition_pos
    specificity = val_TN / val_condition_neg
    MAcc= (sensitivity + specificity) / 2

    if val_pred_pos == 0:
        precision = 0
        F1_score = 0
    else:
        precision = val_TP / val_pred_pos
        F1_score = 2 * (precision*sensitivity)/(precision+sensitivity)
    
    log.update({'sensitivity': sensitivity})
    log.update({'specificity': specificity})
    log.update({'MAcc':MAcc})
    log.update({'F1_score': F1_score})
    logger.info(log)

def evaluation(data_loader, 
               model, 
               loss_fn, 
               metric_fns,
               resume,
               logger):

    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    val_pred_pos = 0
    val_pred_neg = 0
    val_condition_pos = 0
    val_condition_neg = 0
    val_TP = 0
    val_TN = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            
            _,indices = torch.max(output, 1)
            
            pred_pos_idx = [i for i, x in enumerate(indices) if x == 0]
            condition_pos_idx = [i for i, x in enumerate(target) if x == 0]
            pred_neg_idx = [i for i, x in enumerate(indices) if x == 1]
            condition_neg_idx = [i for i, x in enumerate(target) if x == 1]
            
            val_pred_pos += len(pred_pos_idx)
            val_condition_pos += len(condition_pos_idx)
            val_pred_neg += len(pred_neg_idx)
            val_condition_neg += len(condition_neg_idx)
            val_TP += len([x for x in pred_pos_idx if x in condition_pos_idx])
            val_TN += len([x for x in pred_neg_idx if x in condition_neg_idx])
            correct = (indices == target).float()
            val_correct += torch.sum(correct).item()
            val_total += target.shape[0]
            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    sensitivity = val_TP / val_condition_pos
    specificity = val_TN / val_condition_neg
    MAcc= (sensitivity + specificity) / 2

    if val_pred_pos == 0:
        precision = 0
        F1_score = 0
    else:
        precision = val_TP / val_pred_pos
        F1_score = 2 * (precision*sensitivity)/(precision+sensitivity)
    
    log.update({'sensitivity': sensitivity})
    log.update({'specificity': specificity})
    log.update({'MAcc':MAcc})
    log.update({'F1_score': F1_score})
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    test(config)
