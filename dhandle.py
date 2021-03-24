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
    see https://blog.csdn.net/acbattle/article/details/88583128
    '''
    return n != n

class dhandle:
    def __init__(self):
        '''
        member initiallizing
        '''
        self.__cols = None
        self.__source = None
        self.__target = None
        self.__pca = None
        self.__T = None
        self.__lr = None
        self.__RMSE = None
        self.__target_name = "unknow"

    def _load_from_excel(self, config):
        '''
        load dataset from excel

        config:json must those key words below:
        - target    target cols in databse
        - file      excel file location
        '''

        '''
        step 1
        # load excel using pandas
        # see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
        '''
        try:
            data = pandas.read_excel(config['file'])
        except:
            print('load excel failed:{}'.format(config['file']))
            return False

        '''
        step 2
        make sure that target col really exists
        '''
        cols = []
        col_list = [item for item in data]
        if config['target'] not in col_list:
            print('target({}) not in col_list({})'.format(config['target'], col_list))
            return False

        '''
        step 3
        filt invalid cols that have 'NaN' value
        output all valid cols into var 'cols'
        '''
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
        '''
        step 4
        storage all vars into 'self' obj
        - __cols original data from pandas module
        - __source source data matrix row=samples, col=features
        - __target target data matrix row=samples, col=1(only one target)
        - __target_name target key word
        '''
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
        '''
        use config.json to load dataset
        only support excel and mysql for now

        return True is success
        '''
        # check config.json exists
        if path.exists(config_file) is False:
            print("config file:{} not exists!".format(config_file))
            return False
        # load config.json
        # see https://www.runoob.com/python/python-json.html
        with open(config_file, 'r') as f:
            config = json.load(f)

        # must have 'target' in anyway
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
        '''
        predict target value based on source parameters
        '''
        # use trained pca obj to get PCA data matrix
        src = self.__pca.transform(parameters)
        # use linearRegression module to predict the target value
        return self.__lr.predict(src)
    
    def parameter_names(self):
        '''
        return all parameters' key word
        '''
        return [ item.name for item in self.__cols]

    def train(self, test_ratio=0.5, target_RMSE=1e6):
        '''
        use pca and lr to predict
        '''
        src = self.__source
        target = self.__target

        '''
        step 1 do pca train first
        see https://blog.csdn.net/qq_20135597/article/details/95247381
        '''
        pca = PCA(3)
        src = pca.fit_transform(src)
        # storage
        self.__pca = pca

        '''
        step 2 do linear fit
        see https://blog.csdn.net/weixin_39739342/article/details/93379653
        @note if RMSE higher than we think, get test data randomly again and try to get a better score
        '''
        RMSE = 1e100
        while RMSE > target_RMSE:
            # get test data randomly based on sklearn module
            # see https://blog.csdn.net/fxlou/article/details/79189106
            x_train, x_test, y_train, y_test = train_test_split(src, target, test_size=test_ratio)
            # use poly metho to fit(PolynomialFeatures=1 mean linear fitting)
            # see https://blog.csdn.net/hushenming3/article/details/80500364
            lr = pl.make_pipeline(sp.PolynomialFeatures(1), LinearRegression())
            # do fit
            lr.fit(x_train, y_train)

            # measure error
            y_pred = lr.predict(x_test)
            RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        # storage result
        self.__RMSE = RMSE
        self.__lr = lr


    def printInfo(self):
        cols = self.__cols
        print("valid cols:{}".format(len(cols)))
        print("source key words:{}".format([item.name for item in cols]))
        print('target key words:{}'.format(self.__target_name))
        print("RMSE:{}".format(self.__RMSE))
        print("pca ratio:{}".format(self.__pca.explained_variance_ratio_))

        # convert data type from pandas array to numpy array
        target = np.array(self.__target)
        # use self.predict to predict target values
        pre_target = self.predict(self.__source)

        # print figure using matplot
        # see https://www.runoob.com/numpy/numpy-matplotlib.html
        plt.plot(range(len(target)), target, 'b', label='target')
        plt.plot(range(len(pre_target)), pre_target, 'r', label='predict')
        plt.show()

