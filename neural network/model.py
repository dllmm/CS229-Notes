import numpy as np
class nn():
    def __init__(self, config):
        self.hidden_num = config['hidden_num']
        self.neuron_num = config['neuron_num']
        self.alpha = config['alpha']
        self.lamda = config['lamda']
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def theta_init(self, x, y):
        theta = []
        theta.append(np.matrix(np.random.rand(self.neuron_num , x.shape[1] + 1))*0.25)
        for i in range(self.hidden_num - 1):
            theta.append(np.matrix(np.random.rand(self.neuron_num , self.neuron_num + 1))*0.25)
        theta.append(np.matrix(np.random.rand(y.shape[1], self.neuron_num + 1))*0.25)
        return theta
    def loss(self, x, y, theta):
        a = []
        pre_y = x
        a.append(pre_y)
        ls = 0.
        regular = 0.
        for ta in theta:
            a.append(self.sigmoid(np.column_stack((np.matrix(np.ones([pre_y.shape[0], 1])), pre_y)).dot(np.transpose(ta))))
            pre_y = self.sigmoid(np.column_stack((np.matrix(np.ones([pre_y.shape[0], 1])), pre_y)).dot(np.transpose(ta)))
            for i in range(ta.shape[0]):
                regular += ta[i, 1:].dot(np.transpose(ta[i, 1:]))
        for i in range(y.shape[0]):
            ls += ((-y)[i, :].dot(np.transpose(np.log(pre_y[i, :])))) - (np.matrix(np.ones([1, y.shape[1]])) - y[i, :]).dot(
                np.transpose(np.log(np.matrix(np.ones([1, pre_y.shape[1]])) - pre_y[i, :]))
            )
        return ls / x.shape[0] + self.lamda * (regular)/2/x.shape[0], a, pre_y
    def grad (self, x, y, a, theta):
        r = []
        for i in range(len(theta)):
            r.append(self.alpha*self.lamda / x.shape[0] * theta[i][:, 1:])
        for i in range(x.shape[0]):
            for j in range(len(theta)):
                if j == 0:
                    dota = np.multiply(
                        np.multiply(
                        a[len(a) - 1 - j][i, :], np.matrix(np.ones([1, a[len(a) - 1 - j].shape[1]])) - a[len(a) - 1 - j][i, :]),
                        np.multiply(-y[i,:],1/a[len(a) - 1 - j][i, :]) + np.multiply(
                            np.matrix(np.ones([1, y.shape[1]])) - y[i,:], 1/(np.matrix(np.ones([1, a[len(a) - 1 - j].shape[1]])) - a[len(a) - 1 - j][i, :])
                        ))
                else:
                    dota = np.multiply(np.multiply(
                        a[len(a) - 1 - j][i, :], np.matrix(np.ones([1, a[len(a) - 1 - j].shape[1]])) - a[len(a) - 1 - j][i, :]), (theta[len(theta) - j][:,1:].T.dot(dota.T)).T)
                da = dota.T.dot(np.column_stack((np.matrix(np.ones([1,1])),a[len(a) - 2 - j][i, :])))/x.shape[0]
                theta[len(theta) - 1 - j] = theta[len(theta) - 1 - j] - self.alpha * da
        for i in range(len(theta)):
            theta[i][:, 1:] = theta[i][:, 1:] - r[i]
        return theta
















