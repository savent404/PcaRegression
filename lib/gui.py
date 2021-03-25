from PyQt5.QtWidgets import *
import numpy as np


class InputItem(QWidget):
    '''
    a Item like: '[Label]   [InputBox]'
    Use QWidget to bind many widgets together
    see https://blog.csdn.net/qq_38641985/article/details/83377355
    '''

    def __init__(self, label="unknow"):
        super().__init__()

        # use BoxLayout to put item in right position automatically
        # see https://blog.51cto.com/9291927/2423303
        layout = QHBoxLayout()
        # use QLabel to show some text
        # see https://blog.csdn.net/jia666666/article/details/81504595
        self.label = QLabel(label)
        # use QLineEdit to get user input
        # see https://blog.csdn.net/jia666666/article/details/81510502
        self.edit = QLineEdit()
        # see https://blog.csdn.net/qq_25973779/article/details/80001454
        layout.addWidget(self.label)
        # see https://blog.csdn.net/wowocpp/article/details/105264371
        layout.addStretch(1)
        layout.addWidget(self.edit)
        # apply layout
        # see https://blog.csdn.net/zzzzlei123123123/article/details/103764484
        self.setLayout(layout)

    def get_label(self):
        '''
        return the label name
        '''
        return self.label.text()

    def get_input(self):
        '''
        try to return a float number.
        return 'None' if failed
        '''
        try:
            return float(self.edit.text())
        except:
            '''
            use a warning box to notice people that he forget to input every 'QLineEdit'
            or input is not number
            QMessageBox usage, see https://blog.csdn.net/jia666666/article/details/81540785
            '''
            QMessageBox().warning(self, "warning", "{} is not a number".format(self.get_label()))
            return None


class MainWindow(QWidget):
    def __init__(self, input_list=None, predict_func=None):
        super().__init__()

        # predict button
        button = QPushButton("predict")
        # when button been clicked, qt platform will call 'self.on_clicked' function
        # see https://www.jianshu.com/p/6165fec38064
        button.clicked.connect(self.on_clicked)

        # predict result
        # use QLabel to show some text
        # see https://blog.csdn.net/jia666666/article/details/81504595
        label = QLabel("result:")
        # set lable's fixed size
        label.setFixedWidth(200)

        # input parameters
        items = []
        for i in input_list:
            items.append(InputItem(i))

        # create a 1*n layout
        hlayout = QHBoxLayout()
        hlayout.addWidget(button)
        hlayout.addStretch(5)
        hlayout.addWidget(label)

        # create a layout that allow numbers of rows and cols
        vlayout = QGridLayout()
        '''
        put every 'InputBox' automatically
        @note only allow 4 'InputBox' in a row
        @note put predict button and result label in the last slot
        '''
        max_col = 4
        for l in range(0, len(items)):
            col, row = l % max_col, l / max_col
            vlayout.addWidget(items[l], row, col)
        l = l + 1
        col, row = l % max_col, l / max_col
        vlayout.addLayout(hlayout, row, col)

        # apply layout
        self.setLayout(vlayout)

        '''
        storage obj into 'self'
        '''
        # storage predict func
        self.predict = predict_func
        self.items = items
        self.button_predict = button
        self.label_result = label

        self.show()

    def on_clicked(self):
        '''
        get every input value in items into a list
        if the value is invalid,
        'InputBox::get_input()' will throw a warning box
        '''
        list = []
        for item in self.items:
            number = item.get_input()
            if number is None:
                return
            list.append(number)
        # reshape 1D matrix to a 2D matrix(1 row and N cols)
        list = np.reshape(list, (1, -1))
        # use predict function that come from other module
        result = self.predict(list)
        # output result to 'label_result'
        self.label_result.setText("result: {}".format(result))
