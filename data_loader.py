import pandas
import numpy as np
import mysql.connector as connector
from mysql.connector import Error as SQLError


def isInvalidValue(n: float):
    '''
    check number(float type) is NaN or null
    see https://blog.csdn.net/acbattle/article/details/88583128
    '''
    return n != n or n is None


def hasInvalidValue(n):
    '''
    check array contains any 'NaN' or 'null'
    '''
    for i in n:
        if isInvalidValue(i):
            return True
    return False


def from_excel(file="./demo.xlsx", target="k1"):
    '''
    load dataset from excel

    - target    target cols in databse
    - file      excel file location
    '''

    '''
    step 1
    # load excel using pandas
    # see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
    '''
    try:
        data = pandas.read_excel(file)
    except:
        print('load excel failed:{}'.format(file))
        return None

    '''
        step 2
        make sure that target col really exists
        '''
    cols = []
    col_list = [item for item in data]
    if target not in col_list:
        print('target({}) not in col_list({})'.format(
            target, col_list))
        return None

    '''
        step 3
        filt invalid cols that have 'NaN' value
        output all valid cols into var 'cols'
        '''
    height, width = data.shape
    target_index = -1
    for i in range(0, width):
        if target == data[data.columns[i]].name:
            if isInvalidValue(i):
                raise ValueError("target can't has NaN item")
            else:
                target_index = i
                continue
        if hasInvalidValue(data[data.columns[i]]) is False:
            d = data[data.columns[i]]
            cols.append(d)
    '''
    step 4
    storage all vars into 'self' obj
    - feature_data source data matrix row=samples, col=features
    - feature_names source names matrix row=1, col=features
    - target_data target data matrix row=samples, col=1(only one target)
    - target_name key word
    '''
    valid_cols = len(cols)
    source = np.zeros((height, valid_cols), dtype=np.float64)
    for i in range(valid_cols):
        source[:, i] = cols[i].array
    feature_data = source
    feature_names = [item for item in data]
    target_data = data[data.columns[target_index]].array
    return feature_data, feature_names, target_data

def from_mysql(host='localhost', port=3306, user='user', passwd='123', database='database', table='table', target='k1'):
    '''
    load dataset from excel

    config:json must have those key words below:
    - target    target cols in database
    - host      mysql ip
    - port      mysql server port
    - user      mysql user
    - password  mysql user's password
    - database  mysql databse
    - table     mysql table
    '''
    try:
        # try to connect mysql server
        db = connector.connect(
            host=host,
            port=port,
            user=user,
            passwd=passwd,
            database=database,
        )

        # query a table based on config:json
        cursor = db.cursor()
        cursor.execute("SELECT * FROM `{}` where `{}` is not null".format(table, target))
        data = cursor.fetchall()

        # get feature cols describe
        cursor.execute("describe `{}`".format(table))
        features = cursor.fetchall()
        col_names = [item[0] for item in features]

        # get feature number
        cols = len(data[0])
        data = np.reshape(data, (-1, cols))

        if target not in col_names:
            print('target({}) not in col_names({})'.format(
                target, col_names))
            return False
        
        # filt invalid cols that have 'NULL' value
        # output all valid cols into var 'cols'
        arr = []
        arr_names = []
        feature_col = None
        for i in range(0, cols):
            if col_names[i] == target:
                feature_col = data[:,i]
                if hasInvalidValue(feature_col):
                    raise ValueError("target key can't has NaN or NULL")
                continue
            if hasInvalidValue(data[:, i]) is False:
                arr.append(data[:, i])
                arr_names.append(col_names[i])
        feature_names = arr_names
        feature_data = np.reshape(arr, (-1, len(arr)))
        target_data = feature_col
        return feature_data, feature_names, target_data
    except SQLError as e:
        print("SQL error:", type(e), e)
        return None
    except KeyError as e:
        print("invalid config file:", type(e), e)
        return None
    except:
        print("unknow error")
        return None
    