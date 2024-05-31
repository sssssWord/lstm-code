'''

# copy from https://www.zhihu.com/tardis/zm/art/104475016?source_id=1005
# sin prediction
# no legal dataset got

# -*- coding : UTF-8 -*-

'''

# import scipy.io as sio
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from package_LSTM.LSTMRNN import LSTMRNN

INPUT_FEATURES_NUM = 1
OUTPUT_FEATURES_NUM = 1
# TRAIN_DEVICE = "cpu"
TRAIN_DATA_RATIO = 0.6
DATA_LEN = 2000
LOSS_TARGET_PRECISION = 1e-4
LOSS_INFO_PRINT_ITER = 1e2
MAX_EPOCHS = 1000
SAMPLE_RATIO = 0.3

def main() :
    t = np.linspace(0, 12 * np.pi, DATA_LEN)
    print(t)
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    dataset = np.zeros((DATA_LEN, 2))
    dataset[:,0] = sin_t
    dataset[:,1] = cos_t
    dataset = dataset.astype('float32')
    # plot part of the original dataset

    sample_num = (int)(DATA_LEN * SAMPLE_RATIO)
    plt.figure()
    plt.plot(t[0:sample_num], dataset[0:sample_num,0], label ='sin(t)')
    plt.plot(t[0:sample_num], dataset[0:sample_num,1], label = 'cos(t)')
    plt.plot([2.5, 2.5], [-1.3, 0.55], 'r--', label='t = 2.5') # t = 2.5
    plt.plot([6.8, 6.8], [-1.3, 0.85], 'm--', label='t = 6.8') # t = 6.8
    plt.xlabel('t')
    plt.ylim(-1.2, 1.2)
    plt.ylabel('sin(t) and cos(t)')
    plt.legend(loc='upper right')

    # choose dataset for training and testing
    train_data_len = int(DATA_LEN * TRAIN_DATA_RATIO)
    train_x = dataset[:train_data_len, 0]
    train_y = dataset[:train_data_len, 1]
    t_for_training = t[:train_data_len]

    # test_x = train_x
    # test_y = train_y
    test_x = dataset[train_data_len:, 0]
    test_y = dataset[train_data_len:, 1]
    t_for_testing = t[train_data_len:]

    # ----------------- train -------------------
    train_x_tensor = train_x.reshape(-1, 5, INPUT_FEATURES_NUM) # set batch size to 5
    train_y_tensor = train_y.reshape(-1, 5, OUTPUT_FEATURES_NUM) # set batch size to 5
 
    # transfer data to pytorch tensor
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)
    # test_x_tensor = torch.from_numpy(test_x)
 
    lstm_model = LSTMRNN(INPUT_FEATURES_NUM, 16, output_size=OUTPUT_FEATURES_NUM, num_layers=1) # 16 hidden units
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)
 
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

    for epoch in range(MAX_EPOCHS):
        output = lstm_model(train_x_tensor)
        loss = loss_function(output, train_y_tensor)
 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
        if loss.item() < LOSS_TARGET_PRECISION:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, MAX_EPOCHS, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch+1) % LOSS_INFO_PRINT_ITER == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch+1, MAX_EPOCHS, loss.item()))
 
    # prediction on training dataset
    predictive_y_for_training = lstm_model(train_x_tensor)
    predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # torch.save(lstm_model.state_dict(), 'model_params.pkl') # save model parameters to files
 
    # ----------------- test -------------------
    # lstm_model.load_state_dict(torch.load('model_params.pkl'))  # load model parameters from files
    lstm_model = lstm_model.eval() # switch to testing model

    # prediction on test dataset
    test_x_tensor = test_x.reshape(-1, 5, INPUT_FEATURES_NUM) # set batch size to 5, the same value with the training set
    test_x_tensor = torch.from_numpy(test_x_tensor)
 
    predictive_y_for_testing = lstm_model(test_x_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()
 
    # ----------------- plot -------------------
    plt.figure()
    plt.plot(t_for_training, train_x, 'g', label='sin_train')
    plt.plot(t_for_training, train_y, 'b', label='ref_cos_train')
    plt.plot(t_for_training, predictive_y_for_training, 'y--', label='pred_cos_train')

    plt.plot(t_for_testing, test_x, 'c', label='sin_test')
    plt.plot(t_for_testing, test_y, 'k', label='ref_cos_test')
    plt.plot(t_for_testing, predictive_y_for_testing, 'm--', label='pred_cos_test')

    plt.plot([t[train_data_len], t[train_data_len]], [-1.2, 4.0], 'r--', label='separation line') # separation line

    plt.xlabel('t')
    plt.ylabel('sin(t) and cos(t)')
    plt.xlim(t[0], t[-1])
    plt.ylim(-1.2, 4)
    plt.legend(loc='upper right')
    plt.text(14, 2, "train", size = 15, alpha = 1.0)
    plt.text(20, 2, "test", size = 15, alpha = 1.0)

    plt.show()

if __name__ == '__main__' : 
    main()
