import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# read data set & clean data
def read_data(csv_path):
    df = pd.read_csv(csv_path, header=0, sep=',')
    df.loc[df['gender'] == 'Female', 'gender'] = 1
    df.loc[df['gender'] == 'Male', 'gender'] = 2
    df['gender'] = df['gender'].astype('int64')

    df.loc[df['level'] == 'High', 'level'] = 1
    df.loc[df['level'] == 'Middle', 'level'] = 2
    df.loc[df['level'] == 'Low', 'level'] = 3
    df['level'] = df['level'].astype('int64')

    df.loc[df['insurance'] == 0, 'insurance'] = -1
    df['insurance'] = df['insurance'].astype('int64')

    return df


# one hot encoding
def one_hot_data(df, columns):
    return pd.get_dummies(df, columns=columns)


"""
create model like this 
   y = Ax + B
   sample like this:
   y = [1, 1, 1] * [0.1, 0.2, 0.3]T + 0.5
"""


def create_model(input, output, device):
    model = nn.Linear(input, output)
    model.to(device)  # if exist gpu, could update to gpu
    return model


def train(input_data, output_data, model, hyper_parameter):
    input_data = torch.FloatTensor(input_data)
    output_data = torch.FloatTensor(output_data)

    data_size = len(output_data)

    optimizer = optim.SGD(model.parameters(), lr=hyper_parameter.learning_rate)
    model.train()
    for i in range(hyper_parameter.epoch):
        random_index = torch.randperm(data_size)

        for j in range(0, data_size, hyper_parameter.bach_size):
            random_input = input_data[random_index[j: j + hyper_parameter.bach_size]]
            random_output = output_data[random_index[j: j + hyper_parameter.bach_size]]

            optimizer.zero_grad()
            model_result = model(random_input)
            weight = model.weight

            # lagrange svm
            loss = torch.clamp(1 - random_output * model_result.squeeze(), 0)
            loss = torch.mean(loss)  # lagrange factor is means
            loss += hyper_parameter.discount * (weight.squeeze().t() @ weight.squeeze()) / 2.0

            loss.backward()
            optimizer.step()

            print("times: {:4d}, loss: {}".format(i, float(loss)))


# model calculate
def predict_data(x, model):
    return model(x)


class HyperParameter():
    def __init__(self, epoch, bach_size, discount, learning_rate):
        super(HyperParameter, self).__init__()
        self.epoch = epoch
        self.bach_size = bach_size
        self.discount = discount
        self.learning_rate = learning_rate


if __name__ == '__main__':
    hyper_data = HyperParameter(500, 5, 0.01, 0.1)

    df = read_data('datasets/360-sample.csv')
    df = one_hot_data(df, ['gender', 'level'])
    output_data = np.array(df['insurance'].copy())
    input_data = np.array(df.drop('insurance', axis=1).values)
    mean_data = input_data.mean(axis=0)
    std_data = input_data.std(axis=0)
    input_data = (input_data - mean_data) / std_data

    model = create_model(6, 1, torch.device("cpu"))
    train(input_data, output_data, model, hyper_data)
    torch.save(model, "360-model.pkl")

    print(predict_data(torch.FloatTensor((np.array([600, 1, 0, 0, 0, 1]) - mean_data) / std_data), model))
