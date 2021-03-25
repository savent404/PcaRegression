import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sklearn.preprocessing as sp
import sklearn.pipeline as pl
import matplotlib.pyplot as plt
from os import path
import json

# data loader
import lib.data_loader as dl

def valueColor(value):
    '''
    see https://blog.csdn.net/claroja/article/details/79459596
    '''
    if value > 50e5:
        return 'red'
    elif value > 5e5:
        return 'orange'
    elif value > -5e5:
        return 'olive'
    elif value > -50e5:
        return 'yellowgreen'
    else:
        return 'g'


class dhandle:
    def __init__(self):
        '''
        member initiallizing
        '''
        self.debug_info = {}
        self.module = {}
        self.module['lr'] = None
        self.module['pca'] = None
        self.feature_names = None
        self.feature_data = None
        self.target_data = None
        self.target_name = "unknow"

    def _load_from_excel(self, config):
        result = dl.from_excel(config['file'], config['target'])
        if result is None:
            print("load data from excel failed")
            return False
        else:
            self.feature_data = result[0]
            self.feature_names = result[1]
            self.target_data = result[2]
            self.target_name = config['target']
            return True

    def load_from_mysql(self, config):
        result = dl.from_mysql(host=config['host'], port=config['port'], user=config['user'],
                               passwd=config['password'], database=config['database'], table=config['table'], target=config['target'])
        if result is None:
            print("load data from mysql failed")
            return False
        else:
            self.feature_data = result[0]
            self.feature_names = result[1]
            self.target_data = result[2]
            self.target_name = config['target']
            return True

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
            raise ValueError(
                "method:{} not supported".format(config['method']))

    def predict(self, parameters):
        '''
        predict target value based on source parameters
        '''
        # use trained pca obj to get PCA data matrix
        src = self.module['pca'].transform(parameters)
        # use linearRegression module to predict the target value
        return self.module['lr'].predict(src)

    def parameter_names(self):
        '''
        return all parameters' key word
        '''
        return self.feature_names

    def train(self, test_ratio=0.5, target_score=0.99):
        '''
        use pca and lr to predict
        '''
        src = self.feature_data
        target = self.target_data

        '''
        step 1 do pca train first
        see https://blog.csdn.net/qq_20135597/article/details/95247381
        '''
        pca = PCA(3)
        src = pca.fit_transform(src)
        # storage
        self.module['pca'] = pca

        '''
        step 2 do linear fit
        see https://blog.csdn.net/weixin_39739342/article/details/93379653
        @note if RMSE higher than we think, get test data randomly again and try to get a better score
        '''
        SCORE = None
        loop_cnt = 0
        while SCORE is None or SCORE < target_score:
            # get test data randomly based on sklearn module
            # see https://blog.csdn.net/fxlou/article/details/79189106
            x_train, x_test, y_train, y_test = train_test_split(
                src, target, test_size=test_ratio)
            # use poly metho to fit(PolynomialFeatures=1 mean linear fitting)
            # see https://blog.csdn.net/hushenming3/article/details/80500364
            lr = pl.make_pipeline(sp.PolynomialFeatures(1), LinearRegression())
            # do fit
            lr.fit(x_train, y_train)

            # measure error
            y_pred = lr.predict(x_test)
            SCORE = lr.score(x_test, y_test)
            RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

            loop_cnt = loop_cnt + 1
            if (loop_cnt > 1e3):
                target_score = target_score * 0.9
                loop_cnt = 0
                print("target SCORE is too big , adjust to {}...".format(target_score))

        # storage result
        self.debug_info['RMSE'] = RMSE
        self.debug_info['SCORE'] = SCORE
        self.module['lr'] = lr

    def printInfo(self):
        print("valid cols:{}".format(len(self.feature_names)))
        print("source key words:{}".format(self.feature_names))
        print('target key words:{}'.format(self.target_name))
        print("RMSE:{}".format(self.debug_info['RMSE']))
        print('lr score:', self.debug_info['SCORE'])
        print("pca explained variance:", self.module['pca'].explained_variance_)
        print("pca explained variance ratio:{}".format(
            self.module['pca'].explained_variance_ratio_))

        print('pca components:', self.module['pca'].components_)
        print('lr parameters:', vars(self.module['lr'][1]))
        # convert data type from pandas array to numpy array
        target = np.array(self.target_data)
        # use self.predict to predict target values
        pre_target = self.predict(self.feature_data)
        # pca data
        pca_data = self.module['pca'].transform(self.feature_data)

        # print figure using matplot
        # see https://www.runoob.com/numpy/numpy-matplotlib.html


        fig = plt.figure(figsize=plt.figaspect(3,))
        fig.suptitle('')

        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2, projection='3d')
        ax3 = fig.add_subplot(3, 1, 3, projection='3d')

        for i in range(0, len(pre_target)):
            ax2.plot(pca_data[i][0], pca_data[i][1], pca_data[i][2], c=valueColor(pre_target[i]), marker='.')
            ax3.plot(pca_data[i][0], pca_data[i][1], pca_data[i][2], c=valueColor(target[i]), marker='v')
        ax2.set_xlabel('PCA 1th')
        ax2.set_ylabel('PCA 2th')
        ax2.set_zlabel('PCA 3th')

        ax3.set_xlabel('PCA 1th')
        ax3.set_ylabel('PCA 2th')
        ax3.set_zlabel('PCA 3th')

        ax1.plot(range(len(target)), target, 'b', label='target')
        ax1.plot(range(len(pre_target)), pre_target, 'r', label='predict')
        plt.show()

