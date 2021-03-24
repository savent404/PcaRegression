from gui import MainWindow # self defined GUI widget
from dhandle import dhandle # data train/predict obj

import os.path
import sys # system input args
import argparse # parse system input args
import pickle # dump obj to a file
from PyQt5.QtWidgets import QApplication # Qt's GUI platform


def args_setup():
    '''
    setup program arguments
    '''
    parser = argparse.ArgumentParser(description='program for train or predict based on PCA and regression')
    parser.add_argument('--mode', type=str, default='train', dest='method', help='method: train|predict')
    parser.add_argument('--config', type=str, default='./config.json', required=False, dest='config', help='config file if in train mode')
    return parser

if __name__ == '__main__':
    '''
    function entry
    '''
    # get input arguments
    # see https://www.dazhuanlan.com/2019/12/07/5deaed961be69/
    parser = args_setup()
    args = parser.parse_args()

    if args.method == 'train':
        data_handle = dhandle()
        if data_handle.load(args.config) is False:
            print("config file not exist({})!!!\n\n".format(args.config))
            parser.print_help()
            exit(-1)
        data_handle.train()
        data_handle.printInfo()

        # use pickle caching obj to a file
        with open('./.cache', 'wb+') as f:
            pickle.dump(data_handle, f, pickle.HIGHEST_PROTOCOL)
    else :
        if os.path.exists('./.cache') is False:
            print('must train first!!!')
            exit(-1)

        with open('./.cache', 'rb') as f:
            data_handle = pickle.load(f)
        app = QApplication(sys.argv)
        parameter_list = data_handle.parameter_names()
        window = MainWindow(parameter_list, lambda p: data_handle.predict(p))
        sys.exit(app.exec_())
