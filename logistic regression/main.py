import numpy as np
def sigmod(x):
    if type(x) == 'numpy.ndarray':
        xs = []
        for s in x:
            xs.append(1.0/(1.0+np.exp(-s)))
        return np.array(xs)
    else:
        return 1.0/(1.0+np.exp(-x))
def log(x):
    if type(x) == 'numpy.ndarray':
        xs = []
        for s in x:
            xs.append(np.log(s))
        return np.array(xs)
    else:
        return np.log(x)
def read_file(file_path):
    with open(file_path, 'r') as fr:
        lines = fr.readlines()
    dt = []
    for line in lines:
        ld = line.split(' ')
        r = []
        for f in ld:
            if f != '' and f != '\n':
                r.append(f)
        dt.append(r)
    x_train = np.column_stack((np.ones([len(dt), 1], dtype=np.float), np.array(dt)[:, :-1])).astype(float)
    y_train = np.array(dt)[:, -1:].astype(float)
    return x_train, y_train
def loss(x, y, p):
    m = len(x)
    ls = 0.
    for i in range(len(y)):
        if y[i] == 1.:
            ls += -1 * log(sigmod(np.matmul(x[i], p)))
        else:
            ls += -1 * log(1 - sigmod(np.matmul(x[i], p)))
    return ls/m
def gd(x_train,y_train,lear_rate):
    para = np.ones([len(x_train[0]), 1], dtype=np.float)
    ls = loss(x_train, y_train, para[:, 0])
    while True:
        para = para - lear_rate * (np.matmul(np.transpose(x_train), sigmod(
            np.matmul(x_train, para)) - y_train))
        if ls - loss(x_train, y_train, para[:, 0]) < 0.0001:
            break
        else:
            ls = loss(x_train, y_train, para[:, 0])
    return para, ls
if __name__ == '__main__':
    x, y = read_file('dataset.txt')
    p_gd, l_gd = gd(x, y, 0.001)
    print('gd min loss:')
    print(l_gd)
    print('gd parameter:')
    print(p_gd)
    print('train result:')
    print(np.column_stack((sigmod(np.matmul(x, p_gd)),y)))
