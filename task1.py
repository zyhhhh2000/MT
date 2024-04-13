import numpy as np
import sympy as sp
import csv

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QWidget, QPushButton
from PyQt6 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import matplotlib
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('QtAgg')


class MplCanvas3D(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection='3d')
        super(MplCanvas3D, self).__init__(fig)


def count(point1, point2, point3):
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)
    area = 0.5 * np.linalg.norm(np.cross(point2 - point1, point3 - point1))
    return area


def odefun(T, t, c0, eps, c, kij, S_i):
    y1 = T[0]
    y2 = T[1]
    y3 = T[2]
    y4 = T[3]
    y5 = T[4]
    y6 = T[5]
    y7 = T[6]
    y8 = T[7]
    y9 = T[8]
    f1 = (1 / c[0]) * (-kij[0] * (y2 - y1) - eps[0] * S_i[0] * c0 * ((y1 / 100) ** 4))
    f2 = (1 / c[1]) * (-kij[0] * (y2 - y1) - kij[1] * (y3 - y2) - eps[1] * S_i[1] * c0 * ((y2 / 100) ** 4) + (
            22 + 1.0 * np.sin(t / 8)))
    f3 = (1 / c[2]) * (-kij[1] * (y3 - y2) - kij[2] * (y4 - y3) - eps[2] * S_i[2] * c0 * ((y3 / 100) ** 4))
    f4 = (1 / c[3]) * (-kij[2] * (y4 - y3) - kij[3] * (y5 - y4) - eps[3] * S_i[3] * c0 * ((y4 / 100) ** 4))
    f5 = (1 / c[4]) * (-kij[3] * (y5 - y4) - kij[4] * (y6 - y5) - eps[4] * S_i[4] * c0 * ((y5 / 100) ** 4))
    f6 = (1 / c[5]) * (-kij[4] * (y6 - y5) - kij[5] * (y7 - y6) - eps[5] * S_i[5] * c0 * ((y6 / 100) ** 4))
    f7 = (1 / c[6]) * (-kij[5] * (y7 - y6) - kij[6] * (y8 - y7) - eps[6] * S_i[6] * c0 * ((y7 / 100) ** 4))
    f8 = (1 / c[7]) * (-kij[6] * (y8 - y7) - kij[7] * (y9 - y8) - eps[7] * S_i[7] * c0 * ((y8 / 100) ** 4) + (
            22 + 1.0 * np.sin(t / 6)))
    f9 = (1 / c[8]) * (-kij[7] * (y9 - y8) - eps[8] * S_i[8] * c0 * ((y9 / 100) ** 4))
    return [f1, f2, f3, f4, f5, f6, f7, f8, f9]


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.update_count = 0
        self.label1 = QLabel("result.csv")
        self.input1 = QLineEdit()

        self.label2 = QLabel("coefficient.csv")  # 设置标签文本为固定值
        self.input2 = QLineEdit()

        self.label3 = QLabel("start.csv")  # 设置标签文本为固定值
        self.input3 = QLineEdit()

        self.label4 = QLabel("time")  # 设置标签文本为固定值
        self.input4 = QLineEdit()

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label1)
        self.layout.addWidget(self.input1)
        self.layout.addWidget(self.label2)
        self.layout.addWidget(self.input2)
        self.layout.addWidget(self.label3)
        self.layout.addWidget(self.input3)
        self.layout.addWidget(self.label4)
        self.layout.addWidget(self.input4)

        self.button1 = QPushButton("GO!")
        self.layout.addWidget(self.button1)
        self.button2 = QPushButton("PLOT!")
        self.layout.addWidget(self.button2)
        self.button3 = QPushButton("Next")
        self.layout.addWidget(self.button3)

        self.container = QWidget()
        self.container.setLayout(self.layout)

        self.setCentralWidget(self.container)

        self.button1.clicked.connect(self.get_text)
        self.button2.clicked.connect(self.get_text2)
        self.button3.clicked.connect(self.get_text3)

    def update_plot(self):
        self.canvas.axes.cla()

        # 绘制三角形
        norm = Normalize(vmin=np.min(self.sol), vmax=np.max(self.sol))
        cmap = plt.get_cmap('jet')
        sm = ScalarMappable(norm=norm, cmap=cmap)

        self.update_count = self.update_count + 1
        print("Update count:", self.update_count)
        for i in range(9):
            xs = []
            ys = []
            zs = []
            triangles = []
            for element in (self.vertices[i]):
                xs.append(element[0])
                ys.append(element[1])
                zs.append(element[2])
            for element in (self.faces[i]):
                triangles.append([element[0], element[1], element[2]])
            # print(xs)
            # print(triangles)
            self.canvas.axes.plot(np.array(xs), np.array(ys), np.array(zs), color=sm.to_rgba(self.sol[self.update_count][i]))
        self.canvas.draw()

    def load_obj(self, file_path):
        self.vertices = [[] for i in range(9)]
        self.verticess = []
        self.faces = [[] for i in range(9)]
        self.roles = 0
        number = 1

        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('g '):
                    self.roles = self.roles + 1
                elif line.startswith('# object'):
                    number = [int(num) for num in line[-2:-1][0].split()]
                elif line.startswith('v '):
                    vertex = list(map(float, line[2:].split()))
                    self.vertices[number[0] - 1].append((vertex[0], vertex[1], vertex[2]))
                    self.verticess.append((vertex[0], vertex[1], vertex[2]))
                elif line.startswith('f '):
                    indices = list(map(int, line[2:].split()))
                    self.faces[number[0] - 1].append((indices[0], indices[1], indices[2]))
                    # faces[roles-1].append(((vertices[roles-1][indices[0]],vertices[roles-1][indices[1]],vertices[roles-1][indices[2]])))
        # print(vertices)

    def get_text(self):
        text1 = self.input1.text()
        text2 = self.input2.text()
        text3 = self.input3.text()
        self.label1.setText(text1)
        self.label2.setText(text2)
        self.label3.setText(text3)

        self.load_obj('model3.obj')  # 对obj文件进行读取操作
        self.S_i = np.zeros(9)
        self.S_ij = np.zeros((9, 9))

        for i in range(9):
            x = []
            y = []
            z = []
            for element in (self.vertices[i]):
                max = 0
                min = 0
                x.append(element[0])
                z.append(element[1])
                y.append(element[2])
            for element in (self.faces[i]):
                if np.min([self.verticess[element[0] - 1][1], self.verticess[element[1] - 1][1],
                           self.verticess[element[2] - 1][1]]) == np.max(
                    z):
                    max = max + count(self.verticess[element[0] - 1], self.verticess[element[1] - 1],
                                      self.verticess[element[2] - 1])
                elif np.max(
                        [self.verticess[element[0] - 1][1], self.verticess[element[1] - 1][1],
                         self.verticess[element[2] - 1][1]]) == np.min(
                    z):
                    min = min + count(self.verticess[element[0] - 1], self.verticess[element[1] - 1],
                                      self.verticess[element[2] - 1])
            # print(min,max,i,np.min(z),np.max(z))
            if i == 0:
                self.S_ij[i][i + 1] = self.S_ij[i + 1][i] = max
            elif i == 8:
                self.S_ij[i - 1][i] = self.S_ij[i][i - 1] = min
            elif i != 4:
                self.S_ij[i][i + 1] = self.S_ij[i + 1][i] = max
                self.S_ij[i - 1][i] = self.S_ij[i][i - 1] = min
        for i in range(9):
            for element in (self.faces[i]):
                self.S_i[i] = self.S_i[i] + count(self.verticess[element[0] - 1], self.verticess[element[1] - 1],
                                        self.verticess[element[2] - 1])
            # print(S_i[i])
            if i == 0:
                self.S_i[i] = self.S_i[i] - self.S_ij[i][i + 1]
            elif i == 8:
                self.S_i[i] = self.S_i[i] - self.S_ij[i - 1][i]
            else:
                self.S_i[i] = self.S_i[i] - self.S_ij[i - 1][i] - self.S_ij[i][i + 1]
        #
        # print(S_i)
        # print('/')
        # print(S_ij)
        # print('/')

        with open("coefficient.csv", newline='') as csvfile:  # self.input2.text()
            csv_reader = csv.reader(csvfile)
            line = 0
            self.eps = []
            self.c = []
            self.Q = []
            self.lamda = []
            for row in csv_reader:
                for num in row:
                    if line == 0:
                        self.eps.append(float(num))
                    elif line == 1:
                        self.c.append(float(num))
                    elif line == 2:
                        self.Q.append(num)
                    elif line == 3:
                        self.lamda.append(float(num))
                    else:
                        break
                line = line + 1
        self.c0 = 5.67
        self.kij = np.zeros(8)
        for i in range(8):
            self.kij[i] = self.lamda[i] * self.S_ij[i][i + 1]
            
        # init = 1, 1, 1, 1, 1, 1, 1, 1, 1
        with open("start.csv", newline='') as csvfile:  # self.input3.text()
            csv_reader = csv.reader(csvfile)
            init = []
            for row in csv_reader:
                for num in row:
                    init.append(int(num))
        init1 = tuple(init)
        if 3000 >= int(self.input4.text()) > 0:
            t = np.linspace(0, int(self.input4.text()), 1001)
            self.sol = odeint(odefun, init1, t, args=(self.c0, self.eps, self.c, self.kij, self.S_i))
            # print(sol)
            plt.grid()
            plt.plot(t, self.sol[:, 0], color='b', label=r"$T_1$")
            plt.plot(t, self.sol[:, 1], color='c', label=r"$T_2$")
            plt.plot(t, self.sol[:, 2], color='g', label=r"$T_3$")
            plt.plot(t, self.sol[:, 3], color='k', label=r"$T_4$")
            plt.plot(t, self.sol[:, 4], color='m', label=r"$T_5$")
            plt.plot(t, self.sol[:, 5], color='r', label=r"$T_6$")
            plt.plot(t, self.sol[:, 6], color='y', label=r"$T_7$")
            plt.plot(t, self.sol[:, 7], color='pink', label=r"$T_8$")
            plt.plot(t, self.sol[:, 8], color='tan', label=r"$T_9$")
            plt.legend(loc="best")
            plt.show()

            data = pd.DataFrame(
                {'t={}'.format(int(self.input4.text())): t, 'T1': self.sol[:, 0], 'T2': self.sol[:, 1],
                 'T3': self.sol[:, 2],
                 'T4': self.sol[:, 3], 'T5': self.sol[:, 4],
                 'T6': self.sol[:, 5],
                 'T7': self.sol[:, 6], 'T8': self.sol[:, 7], 'T9': self.sol[:, 8]})
            data.to_csv("result.csv", index=False, sep=',')  # self.input1.text()
        else:
            self.bigbig = int(self.input4.text())-10
            t = np.linspace(0, 10, 1001)
            self.sol = odeint(odefun, init1, t, args=(self.c0, self.eps, self.c, self.kij, self.S_i))
            # print(sol)
            plt.grid()
            plt.plot(t, self.sol[:, 0], color='b', label=r"$T_1$")
            plt.plot(t, self.sol[:, 1], color='c', label=r"$T_2$")
            plt.plot(t, self.sol[:, 2], color='g', label=r"$T_3$")
            plt.plot(t, self.sol[:, 3], color='k', label=r"$T_4$")
            plt.plot(t, self.sol[:, 4], color='m', label=r"$T_5$")
            plt.plot(t, self.sol[:, 5], color='r', label=r"$T_6$")
            plt.plot(t, self.sol[:, 6], color='y', label=r"$T_7$")
            plt.plot(t, self.sol[:, 7], color='pink', label=r"$T_8$")
            plt.plot(t, self.sol[:, 8], color='tan', label=r"$T_9$")
            plt.legend(loc="best")
            plt.show()

            data = pd.DataFrame(
                {'t={}'.format(int(self.input4.text())): t, 'T1': self.sol[:, 0], 'T2': self.sol[:, 1],
                 'T3': self.sol[:, 2],
                 'T4': self.sol[:, 3], 'T5': self.sol[:, 4],
                 'T6': self.sol[:, 5],
                 'T7': self.sol[:, 6], 'T8': self.sol[:, 7], 'T9': self.sol[:, 8]})
            data.to_csv("result.csv", index=False, sep=',')  # self.input1.text()
            self.initt = [self.sol[:, 0][-1],self.sol[:, 1][-1],self.sol[:, 2][-1],self.sol[:, 3][-1],self.sol[:, 4][-1],self.sol[:, 5][-1],self.sol[:, 6][-1],self.sol[:, 7][-1],self.sol[:, 8][-1]]


    def get_text2(self):
        self.canvas = MplCanvas3D(self, width=5, height=4, dpi=100)
        self.setCentralWidget(self.canvas)
        print("1")
        self.show()

        if hasattr(self, 'timer'):  # 检查是否已经存在定时器
            self.timer.stop()  # 如果存在，停止之前的定时器

        self.update_plot()
        # print(1)
        # self.timer.setInterval(1000)
        # self.timer.timeout.connect(self.update_plot)
        # self.timer.start()

    def get_text3(self):
        if 3000 < int(self.input4.text()) and self.bigbig>=10:
            self.bigbig = self.bigbig-10
            init1 = self.initt
            print(init1)
            t = np.linspace(0, 10, 1001)
            self.sol = odeint(odefun, init1, t, args=(self.c0, self.eps, self.c, self.kij, self.S_i))
            # print(sol)
            m = int(self.input4.text())-self.bigbig
            xtik = np.linspace(m-10, m, 11)
            xtikk = np.linspace(0, 10, 11)
            print(xtik)
            plt.close()
            plt.grid()
            plt.plot(t, self.sol[:, 0], color='b', label=r"$T_1$")
            plt.plot(t, self.sol[:, 1], color='c', label=r"$T_2$")
            plt.plot(t, self.sol[:, 2], color='g', label=r"$T_3$")
            plt.plot(t, self.sol[:, 3], color='k', label=r"$T_4$")
            plt.plot(t, self.sol[:, 4], color='m', label=r"$T_5$")
            plt.plot(t, self.sol[:, 5], color='r', label=r"$T_6$")
            plt.plot(t, self.sol[:, 6], color='y', label=r"$T_7$")
            plt.plot(t, self.sol[:, 7], color='pink', label=r"$T_8$")
            plt.plot(t, self.sol[:, 8], color='tan', label=r"$T_9$")
            plt.xticks(xtikk,xtik)
            plt.legend(loc="best")
            plt.show()

            data = pd.DataFrame(
                {'t={}'.format(int(m)): t, 'T1': self.sol[:, 0], 'T2': self.sol[:, 1],
                 'T3': self.sol[:, 2],
                 'T4': self.sol[:, 3], 'T5': self.sol[:, 4],
                 'T6': self.sol[:, 5],
                 'T7': self.sol[:, 6], 'T8': self.sol[:, 7], 'T9': self.sol[:, 8]})
            data.to_csv("result.csv", index=False, sep=',')  # self.input1.text()
            self.initt = [self.sol[:, 0][-1],self.sol[:, 1][-1],self.sol[:, 2][-1],self.sol[:, 3][-1],self.sol[:, 4][-1],self.sol[:, 5][-1],self.sol[:, 6][-1],self.sol[:, 7][-1],self.sol[:, 8][-1]]




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()
