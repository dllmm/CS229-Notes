import numpy as np
import csv
def read_file(file_path, tzsf):
    cf = csv.reader(open(file_path,'r'))
    dt = list(cf)
    x_train = np.column_stack((np.ones([len(dt) - 1, 1], dtype=np.float), np.array(dt)[1:, :3].astype(float)))
    x_tzsf = np.column_stack((np.ones([len(dt) - 1, 1], dtype=np.float), np.array(dt)[1:, :3].astype(float)/tzsf))
    y_train = np.array(dt)[1:,-1:].astype(float)
    y_tzsf = np.array(dt)[1:,-1:].astype(float)/tzsf
    return x_train,y_train,x_tzsf, y_tzsf
def loss_gd(x_train,y_train,tzsf):
    lear_rate = 0.2
    para = np.zeros([len(x_train[0]),1],dtype=np.float)
    ls = 1/2/len(x_train)*(
        np.matmul(np.transpose(np.matmul(x_train,para) - y_train),np.matmul(x_train,para) - y_train)[0][0])
    while True:
        para = para - lear_rate/len(x_train)*(np.matmul(np.transpose(x_train),np.matmul(x_train,para) - y_train))
        if ls  == 1 / 2 / len(x_train) * (
        np.matmul(np.transpose(np.matmul(x_train, para) - y_train), np.matmul(x_train, para) - y_train)[0][0]) :
            break
        else:
            ls = 1 / 2 / len(x_train) * (
                np.matmul(np.transpose(np.matmul(x_train, para) - y_train), np.matmul(x_train, para) - y_train)[0][0])
    return np.row_stack((para[0]*tzsf,para[1:])),ls * tzsf * tzsf
def loss_ne(x,y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x)),np.transpose(x)),y),1 / 2 / len(x) * (
                np.matmul(np.transpose(np.matmul(x, np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x)),np.transpose(x)),y)) - y), np.matmul(x, np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x)),np.transpose(x)),y)) - y)[0][0])
if __name__ == '__main__':
    x,y,x_tzsf,y_tzsf = read_file('50_Startups.csv',100000)
    p_gd,l_gd = loss_gd(x_tzsf,y_tzsf,100000)
    p_ne, l_ne = loss_ne(x, y)
    print('gd min loss:')
    print(l_gd)
    print('gd parameter:')
    print(p_gd)
    print('ne min loss:')
    print(l_ne)
    print('ne parameter:')
    print(p_ne)
