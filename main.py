import pandas
import numpy as np
from sklearn.decomposition import PCA

def isNaN(n):
    '''
    check number(float type) is NaN
    '''
    return n != n

class dhandle:
    def __init__(self, file):
        data = pandas.read_excel(file)
        cols = []

        # drop col if has 'NaN'
        height, width = data.shape
        for i in range(0, width):
            valid = True
            for j in data[data.columns[i]]:
                if isNaN(j):
                    valid = False
                    break
            if valid is True:
                d = data[data.columns[i]]
                cols.append(d)
        
        valid_cols = len(cols)
        print("valid cols:{}".format(valid_cols))
        data = np.zeros((height, valid_cols), dtype=np.float)
        for i in range(valid_cols):
            # data[:, i] = np.array(cols[i].array, dtype=np.float)
            data[:, i] = cols[i].array
        
        print("input matrix: {}".format(data))
        self.__cols = cols
        self.__data = data

    def doPCA(self):
        pca = PCA(n_components = 1)
        
        d = pca.fit_transform(self.__data)
        # inv_d = pca.inverse_transform(self.__data)

        print(pca.explained_variance_ratio_)
        pass

if __name__ == '__main__':
    d = dhandle('./d1.xlsx')
    d.doPCA()
