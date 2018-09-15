from nn import NeuraNetwork
from sklearn import datasets
import matplotlib.pyplot as plt

n = NeuraNetwork(64,1000,10)
digits = datasets.load_digits()

x, y = digits.data[:], digits.target[:]

for k in range(100):
	for i in range(100):
		temp = [0 for i in range(10)]
		index = y[i]
		temp[index] = 1
		n.backward(x[i], temp)


print(n.loss)
for i in range(0,10):
	out = n.feedForward(x[i])
	print(f"expected {y[i]}, got {out.index(max(out))}")
