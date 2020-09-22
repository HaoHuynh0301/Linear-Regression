import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot
import math
import sklearn

weight=0.03
bias=0.001
LEARNING_RATE=0.001
ITERS=60

dataset=pd.read_csv('advertising.csv', header = 0)
data_value=dataset.values[:, 2]
data_target=dataset.values[:, 4]

# pyplot.scatter(data_value, data_target, marker = 'o')
# pyplot.show()

def predict(radio, weight, bias):
    return weight*radio+bias

def cost_function(data_value, data_target, weight, bias):
    n=len(data_value)
    sum=0
    for i in range(n):
        sum+=((data_target[i]-(weight*data_value[i]+bias))**2)

    return sum/n

def update_weight_bias(data_value, data_target, weight, bias, learning_rate):
    n=len(data_value)
    weight_temp=0.0
    bias_temp=0.0
    for i in range(n):
        weight_temp+=(-2*data_value[i]*(data_target[i]-(weight*data_value[i]+bias)))
        bias_temp += (-2 * (data_target[i] - (weight * data_value[i] + bias)))

    weight-=(weight_temp/n)*learning_rate
    bias-=(bias_temp/n)*learning_rate

    return  weight, bias

def train(data_value, data_target, weight, bias, learning_rate, iters):
    cost=[]
    for i in range(iters):
        weight, bias=update_weight_bias(data_value, data_target, weight, bias, learning_rate)
        cost_temp=cost_function(data_value, data_target, weight, bias); cost.append(cost_temp)

    return weight, bias, cost

new_weight, new_bias, cost=train(data_value, data_target, weight, bias, LEARNING_RATE, ITERS)

print(cost)

predict_result=predict(22, new_weight, new_bias); print(predict_result)

print(dataset)



