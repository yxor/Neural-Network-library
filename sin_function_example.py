from nn import NeuraNetwork
from math import sin, pi
import matplotlib.pyplot as plt
import random

nn = NeuraNetwork(1,10,1)
DATASET_SIZE = 10000



def make_dataset(number, function):
	dataset = []
	for i in range(number):
		inputs = [random.uniform(0,1)]
		outputs = [(function(inputs[0]*2*pi)+1)/2]
		dataset.append([inputs,outputs])
	return dataset

def graph_it(color):
	for i in range(100):
		somethin = 0.01*i
		out = nn.feedForward([somethin])
		plt.scatter(*[somethin,out[0]], c=color)

#graphing the output of the untrained nn
graph_it("g")

#trainging the nn with different datasets and graphing it as it improves
dataset = make_dataset(DATASET_SIZE)
for data in dataset:
	nn.backward(data[0],data[1])
graph_it("b")
dataset = make_dataset(DATASET_SIZE)
for data in dataset:
	nn.backward(data[0],data[1])
graph_it("r")

dataset = make_dataset(DATASET_SIZE)
for data in dataset:
	nn.backward(data[0],data[1])
graph_it("c")

dataset = make_dataset(DATASET_SIZE)
for data in dataset:
	nn.backward(data[0],data[1])
graph_it("k")

plt.ylabel("sin wave")
plt.show()