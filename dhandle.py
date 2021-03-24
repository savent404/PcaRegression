import pandas
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sklearn.preprocessing as sp
import sklearn.pipeline as pl
import matplotlib.pyplot as plt
from os import error, path
import json

def isNaN(n):
    '''
    check number(float type) is NaN
    '''
    return n != n

use_pca = True
class dhandle:
    def __init__(self):
        self.__cols = None
        self.__source = None
        self.__target = None
        self.__pca = None
        self.__T = None
        self.__lr = None
        self.__RMSE = None
        self.__target_name = "unknow"

    def _load_from_excel(self, config):
        # load excel
        try:
            data = pandas.read_excel(config['file'])
        except:
            print('load excel failed:{}'.format(config['file']))
            return False
        cols = []
        # locate target col first
        col_list = [item for item in data]
        if config['target'] not in col_list:
            print('target({}) not in col_list({})'.format(config['target'], col_list))
            return False
        # drop col if has 'NaN'
        height, width = data.shape
        target_index = -1
        for i in range(0, width):
            if config['target'] == data[data.columns[i]].name:
                if isNaN(i):
                    raise ValueError("target can't has NaN item")
                else:
                    target_index = i
                    continue
            valid = True
            for j in data[data.columns[i]]:
                if isNaN(j):
                    valid = False
                    break
            if valid is True:
                d = data[data.columns[i]]
                cols.append(d)
        # load to self obj
        valid_cols = len(cols)
        source = np.zeros((height, valid_cols), dtype=np.float64)
        for i in range(valid_cols):
            source[:, i] = cols[i].array
        self.__cols = cols
        self.__source = source
        self.__target = data[data.columns[target_index]].array
        self.__target_name = config['target']
        return True
 
    def load_from_mysql(self, config):
        raise NotImplemented

    def load(self, config_file):
        if path.exists(config_file) is False:
            print("config file:{} not exists!".format(config_file))
            return False
        
        with open(config_file, 'r') as f:
            config = json.load(f)

        if config['target'] is None:
            print("config must have 'target'")
            return False
        
        if config['method'] == 'excel':
            return self._load_from_excel(config)
        elif config['method'] == 'mysql':
            return self.load_from_mysql(config)
        else:
            raise ValueError("method:{} not supported".format(config['method']))
        

    def predict(self, parameters):
        if use_pca:
            src = self.__pca.transform(parameters)
        else:
            src = parameters
        return self.__lr.predict(src)
    
    def parameter_names(self):
        return [ item.name for item in self.__cols]

    def train(self, test_ratio=0.5, target_RMSE=1e6):
        src = self.__source
        target = self.__target

        '''
        step 1 do pca fit first
        '''
        if use_pca == True:
            pca = PCA(3)
            src = pca.fit_transform(src)
            # storage
            self.__pca = pca
        else:
            src = src

        '''
        step 2 do linear fit
        '''
        RMSE = 1e100
        while RMSE > target_RMSE:
            x_train, x_test, y_train, y_test = train_test_split(src, target, test_size=test_ratio)
            lr = pl.make_pipeline(sp.PolynomialFeatures(1), LinearRegression())
            lr.fit(x_train, y_train)
            # print(lr.coef_, lr.intercept_)

            '''
            step 3 measure error
            '''
            y_pred = lr.predict(x_test)
            sub = y_pred - y_test
            MSE = metrics.mean_squared_error(y_test, y_pred)
            RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        self.__RMSE = RMSE
        self.__lr = lr


    def printInfo(self):
        cols = self.__cols
        print("valid cols:{}".format(len(cols)))
        print("source key words:{}".format([item.name for item in cols]))
        print('target key words:{}'.format(self.__target_name))
        print("RMSE:{}".format(self.__RMSE))
        if use_pca == True:
            print("pca ratio:{}".format(self.__pca.explained_variance_ratio_))
            src = self.__pca.transform(self.__source)
        else:
            src = self.__source
        target = np.array(self.__target)
        pre_target =  self.__lr.predict(src)

        plt.plot(range(len(target)), target, 'b', label='target')
        plt.plot(range(len(pre_target)), pre_target, 'r', label='predict')
        plt.show()

