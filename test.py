import argparse
import torch
from tqdm import tqdm
import GNN.data_loader as module_data
import GNN.model as module_arch
from parse_config import ConfigParser
import pandas as pd 


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    targets = dict()
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output = model(data)
            for i in range(len(data.y)):
                targets[data.y[i].item()] = output[i].tolist()

    df_targets = pd.DataFrame.from_dict(targets, orient="index")
    df_targets.columns = ['property_%d' % x for x in range(12)]
    df_targets.sort_index(inplace=True)
    df_targets.to_csv('targets.csv', index_label='gdb_idx')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='alchemy_project')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)
