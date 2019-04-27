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
        theta.append(np.matrix(np.random.rand(self.neuron_num , x.shape[1] + 1)))
        for i in range(self.hidden_num - 1):
            theta.append(np.matrix(np.random.rand(self.neuron_num , self.neuron_num + 1)))
        theta.append(np.matrix(np.random.rand(y.shape[1], self.neuron_num + 1)))
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
        delta = []
        d = []
        for j in range(len(theta)):
            da = np.matrix(np.zeros([theta[len(theta) - 1 - j].shape[0],theta[len(theta) - 1 - j].shape[1]]))
            for i in range(x.shape[0]):
                if j == 0:
                    dota = np.multiply(
                        np.multiply(
                        a[len(a) - 1 - j][i, :], np.matrix(np.ones([1, a[len(a) - 1 - j].shape[1]])) - a[len(a) - 1 - j][i, :]),
                        np.multiply(-y[i,:],1/a[len(a) - 1 - j][i, :]) + np.multiply(
                            np.matrix(np.ones([1, y.shape[1]])) - y[i,:], 1/(np.matrix(np.ones([1, a[len(a) - 1 - j].shape[1]])) - a[len(a) - 1 - j][i, :])
                        ))
                    d.append(dota)
                else:
                    dota = np.multiply(np.multiply(
                        a[len(a) - 1 - j][i, :], np.matrix(np.ones([1, a[len(a) - 1 - j].shape[1]])) - a[len(a) - 1 - j][i, :]), (
                        theta[len(theta) - j][:,1:].T.dot(d[i].T)).T)
                    d[i] = dota
                da = da + dota.T.dot(np.column_stack((np.matrix(np.ones([1,1])),a[len(a) - 2 - j][i, :])))/x.shape[0]
            da[:, 1:] += self.lamda / x.shape[0] * theta[len(theta) - 1 - j][:, 1:]
            delta.append(da)
        for k in range(len(theta)):
            theta[k] = theta[k] - self.alpha * delta[len(delta) - 1 - k]
        return theta
















