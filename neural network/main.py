import numpy as np
import csv
from model import nn
def read_file(file_path):
    cf = np.array(list(csv.reader(open(file_path,'r'))))[1:, 1:]
    labels = []
    yy = []
    for label in cf[:, -1]:
        if label not in labels:
            labels.append(label)
    x = np.matrix(cf[:, :-1]).astype('float')
    for label in cf[:, -1]:
        yyy = []
        for i in range(len(labels)):
            if label == labels[i]:
                for j in range(i):
                    yyy.append(0.)
                yyy.append(1.)
                for k in range(len(labels) - 1 - i):
                    yyy.append(0.)
        yy.append(yyy)
    y = np.matrix(yy).astype('float')
    return x, y
if __name__ == '__main__':
    x, y = read_file('iris.csv')
    x_test = np.row_stack((x[40:50,:],x[90:100,:],x[140:,:]))
    y_test = np.row_stack((y[40:50, :], y[90:100, :], y[140:, :]))
    x = np.row_stack((x[0:40, :], x[50:90, :], x[100:140, :]))
    y = np.row_stack((y[0:40, :], y[50:90, :], y[100:140, :]))
    config = {}
    config['hidden_num'] = 1
    config['neuron_num'] = 5
    config['alpha'] = 1
    config['lamda'] = 1
    mynn = nn(config)
    theta = mynn.theta_init(x, y)
    ls, a, pre_y = mynn.loss(x, y, theta)
    for i in range(5000):
        theta = mynn.grad(x, y, a, theta)
        lss,a,pre_y = mynn.loss(x,y,theta)
        if lss > ls:
            mynn.alpha = mynn.alpha/2
        ls = lss
    y_pred = np.array(np.argmax(pre_y, axis=1) + 1)
    y = np.array(np.argmax(y, axis=1) + 1)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('train accuracy = {0}%'.format(accuracy * 100))
    lss_test, a_test, pre_y_test = mynn.loss(x_test, y_test, theta)
    print(pre_y_test)
    print(y_test)
    y_pred_test = np.array(np.argmax(pre_y_test, axis=1) + 1)
    y_test = np.array(np.argmax(y_test, axis=1) + 1)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred_test, y_test)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('test accuracy = {0}%'.format(accuracy * 100))