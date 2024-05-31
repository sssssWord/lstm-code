'''

# copy from https://blog.csdn.net/ting_qifengl/article/details/113039454
# IGBT degeneration prediction
# no legal dataset got

# -*- coding : UTF-8 -*-

'''

# import scipy.io as sio
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from package_LSTM.LSTMRNN import LSTMRNN
# import matplotlib
# import math
# import csv

INPUT_FEATURES_NUM = 53
OUTPUT_FEATURES_NUM = 1
TRAIN_DEVICE = "cpu"
TRAIN_DATA_RATIO = 0.8
DATA_X_PATH = ""    # dataset set to be private
DATA_Y_PATH = ""
LOSS_TARGET_PRECISION = 1e-4
LOSS_INFO_PRINT_ITER = 1e2

def main() :
    device = torch.device(TRAIN_DEVICE)

    # if(torch.cuda.is_available()) : 
    #     device = torch.device("cuda:0")
    #     print("Training on GPU...")
    # else :
    #     print("No legal GPU available, training on CPU...")

    data_x = np.array(pd.read_csv(DATA_X_PATH, header = None)).astype('float32')
    data_y = np.array(pd.read_csv(DATA_Y_PATH, header = None)).astype('float32')

    data_len = len(data_x)
    t = np.linspace(0, data_len, data_len + 1)
    train_data_len = int(data_len * TRAIN_DATA_RATIO)

    train_x = data_x[5 : train_data_len]
    train_y = data_y[5 : train_data_len]
    t_for_training = t[5 : train_data_len]

    test_x = data_x[train_data_len : ]
    test_y = data_y[train_data_len : ]
    t_for_testing = t[train_data_len : ]

    train_x_tensor = train_x.reshape(5, -1, INPUT_FEATURES_NUM)
    train_y_tensor = train_y.reshape(1, OUTPUT_FEATURES_NUM)

    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)

    #---------- train ---------- #

    lstm = LSTMRNN(INPUT_FEATURES_NUM, 20, output_size = OUTPUT_FEATURES_NUM, num_layers = 1)
    
    # debug information : 
    print("lstm : ", lstm, sep = '', end = '\n')
    print("lstm.parameters : ", lstm.parameters, sep = '', end = '\n')
    print("train x tensor dimension : ", Variable(train_x_tensor).size())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr = 1e-2)

    pre_loss = 1e3
    max_iter = 2e3

    train_x_tensor = train_x_tensor.to(device)

    for iter in range(max_iter) : 
        output = lstm(train_x_tensor).to(device)
        loss = criterion(output, train_y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < pre_loss : 
            # save model parameters to file
            torch.save(lstm.state_dict(), 'lstm.pt')
            pre_loss = loss

        if loss.item() < LOSS_TARGET_PRECISION : 
            print("Epoch [{}/{}], loss : {.5f}".format(iter + 1, max_iter, loss.item()))
            print("Loss meets target precision...")
            break
        elif (iter + 1) % LOSS_INFO_PRINT_ITER == 0 : 
            print("Epoch [{}/{}], loss : {.5f}".format(iter + 1, max_iter, loss.item()))

    # prediction on training dataset
    pred_y_for_train = lstm(train_x_tensor).to(device)
    pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # ---------- test ---------- #
    lstm = lstm.eval()

    # prediction on test dataset
    test_x_tensor = test_x.reshape(5, -1, InterruptedError)
    test_x_tensor = torch.from_numpy(test_x_tensor)
    test_x_tensor = test_x_tensor.to(device)

    pred_y_for_test = lstm(test_x_tensor).to(device)
    pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y))
    print("test loss : ", loss.item(), sep = '')

    # ---------- pic ---------- #
    plt.figure()
    plt.plot(t_for_training, train_y, 'b', label = 'y_train')
    plt.plot(t_for_training, pred_y_for_train, 'y--', label = 'pre_train')

    plt.plot(t_for_testing, test_y, 'k', label = 'y_test')
    plt.plot(t_for_testing, pred_y_for_test, 'm--', label = 'pre_test')

    plt.xlabel('x')         # t
    plt.ylabel('y')         # Vce
    plt.show()

if __name__ == '__main__' : 
    main()
