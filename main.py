import pandas
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sklearn.preprocessing as sp
import sklearn.pipeline as pl
import matplotlib.pyplot as plt

def isNaN(n):
    '''
    check number(float type) is NaN
    '''
    return n != n

use_pca = True
class dhandle:

    def __init__(self, file, target_index = 0):
        data = pandas.read_excel(file)
        cols = []

        # drop col if has 'NaN'
        height, width = data.shape
        assert(width > target_index)
        assert(isNaN(data.columns[target_index]) == False)
        for i in range(0, width):
            if target_index == i:
                continue
            valid = True
            for j in data[data.columns[i]]:
                if isNaN(j):
                    valid = False
                    break
            if valid is True:
                d = data[data.columns[i]]
                cols.append(d)

        valid_cols = len(cols)
        source = np.zeros((height, valid_cols), dtype=np.float64)
        for i in range(valid_cols):
            source[:, i] = cols[i].array

        self.__cols = cols
        self.__source = source
        self.__target = data[data.columns[target_index]].array
        self.__pca = None
        self.__T = None
        self.__lr = None

    def train(self, test_ratio=0.5):
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
        RMSE = 1e10
        while RMSE > 10e5:
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
            print("MES:{}, RMES:{}".format(MSE, RMSE))

        self.__lr = lr


    def printInfo(self):
        cols = self.__cols
        print("valid cols:{}".format(len(cols)))
        print("key words:{}".format([item.name for item in cols]))

        if use_pca == True:
            print("variance_ratio:{}".format(self.__pca.explained_variance_ratio_))
            src = self.__pca.transform(self.__source)
        else:
            src = self.__source
        target = np.array(self.__target)
        pre_target =  self.__lr.predict(src)

        plt.plot(range(len(target)), target, 'b', label='target')
        plt.plot(range(len(pre_target)), pre_target, 'r', label='predict')
        plt.show()

if __name__ == '__main__':
    d = dhandle('./d2.xlsx', target_index=57)
    d.train()
    d.printInfo()
