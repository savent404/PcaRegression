from PyQt5.QtCore import qWarning
from PyQt5.QtWidgets import *
import numpy as np
import sys

class InputItem(QWidget):
    def __init__(self, label = "unknow"):
        super().__init__()
        layout = QHBoxLayout()
        self.label = QLabel(label)
        self.edit = QLineEdit()
        layout.addWidget(self.label)
        layout.addStretch(1)
        layout.addWidget(self.edit)
        self.setLayout(layout)

    def get_label(self):
        return self.label.text()

    def get_input(self):
        try:
            return float(self.edit.text())
        except:
            QMessageBox().warning(self, "warning", "{} is not a number".format(self.get_label()))
            return None
        

class MainWindow(QWidget):
    def __init__(self, input_list=None, predict_func=None):
        super().__init__()

        # storage predict func
        self.predict = predict_func

        # predict button
        self.button_predict = QPushButton("predict")
        self.button_predict.clicked.connect(self.on_clicked)

        # predict result
        self.label_result = QLabel("result:")
        self.label_result.setFixedWidth(200)

        # input parameters
        items = []
        for i in input_list:
            items.append(InputItem(i))
        self.items = items

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.button_predict)
        hlayout.addStretch(5)
        hlayout.addWidget(self.label_result)

        vlayout = QGridLayout()
        max_col = 4
        for l in range(0, len(items)):
            col, row = l % max_col, l / max_col
            vlayout.addWidget(items[l], row, col)
        l = l + 1
        col, row = l % max_col, l / max_col
        vlayout.addLayout(hlayout, row, col)
        self.setLayout(vlayout)
        self.show()

    def on_clicked(self):
        list = []
        for item in self.items:
            number = item.get_input()
            if number is None:
                break
            list.append(number)
        list = np.reshape(list, (1, -1))
        result = self.predict(list)
        self.label_result.setText("result: {}".format(result))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(["a", "b", "c"])
    sys.exit(app.exec_())



