from src.model import model
from argparse import ArgumentParser
from config import config


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--file_dir', dest='file_path',
                        help='the directory path of data set',
                        required=False, default='./data')
    parser.add_argument('--log_dir', dest='log_path',
                        help='the directory path to save the tensorflow log files',
                        required=False, default='./log')
    return parser


if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()
    config['data_dir'] = options.file_path
    m = model()
    m.train(options.log_path)

