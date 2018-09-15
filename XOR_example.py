from nn2 import NeuraNetwork
from numpy import add, array
import matplotlib.pyplot as plt
import random


def make_dataset(number):
	dataset = []
	for i in range(number):
		temp = [random.randint(0,1),random.randint(0,1)]
		#if x != y , label the data with 1
		if temp[0] != temp[1]:
			dataset.append([temp,1])
		#if x == y , label the data with 0
		else:
			dataset.append([temp,0])
	return dataset


#initiating an instance of the NeuralNetwork class
#nn = NeuraNetwork(2,8,4,1)
nn = NeuraNetwork(2,8,1)



#creating the dataset
dataset = make_dataset(20000)

#training the nn with the data
for data in dataset:
	nn.backward(data[0],[data[1]])


for i in range(0,2):
	for j in range(0,2):
		out = nn.feedForward([i, j])
		print(f"A:{i} and B:{j} => A XOR B : {out[0]}")

print(f"loss :{nn.loss}")


