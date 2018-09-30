import numpy as np
import random



class NeuraNetwork:

	def __init__(self, *args, **kwargs):
		self.input_nodes = args[0]
		self.output_nodes = args[-1]
		self.hidden_nodes = args[1:-1]
		self.layers = len(args)
		self.nodes = args
		self.weights = []
		self.bias = []
		for i in range(self.layers-1):
			#random weights
			self.weights.append(np.array([[random.uniform(-1,1) for j in range(self.nodes[i])] for k in range(self.nodes[i+1])]))
		for i in range(self.layers-1):
			#random biases
			self.bias.append(np.array([random.uniform(-1,1) for i in range(self.nodes[i+1])]))

		self.hidden_temp = []
		self.output_temp = 0
		self.learningRate = 0.05
		self.loss = None
		
		
	def feedForward(self,inputs):

		if len(inputs) != self.input_nodes:
			raise Exception("Wrong number of inputs, try again with {self.input_nodes} inputs".format(**locals()))

		self.layers_temp = []
		inputu = np.array(inputs)
		self.layers_temp.append(inputu)
		for i in range(self.layers-1):
			hidden = self.weights[i].dot(inputu)
			hidden = np.add(self.bias[i],hidden)
			hidden = self.activation(hidden)
			self.layers_temp.append(hidden)
			inputu = hidden
		self.output_temp = self.layers_temp[-1]
		return list(self.output_temp)
	
	# Activation function: Sigmoid
	def activation(self, m):
		return 1 / (1 + np.exp(-m))
	
	
	def disactivate(self, m):
		return m * (1 - m)

	
	def disigmoid(self,x):
		return np.array([self.disactivate(xi) for xi in x])

	#training the network
	def backward(self,inputs,targets):
		#exception checking
		if len(inputs) != self.input_nodes or len(targets) != self.output_nodes:
			raise Exception("Wrong number of inputs or desired outputs, try again with {self.input_nodes} inputs and {self.output_nodes} outputs.".format(**locals()))
		#initialing the layers 
		self.feedForward(inputs)
		targets = np.array(targets)
		layers = self.layers_temp
		outputs = self.output_temp
		errors =  targets - outputs
		#| || || |_
		if self.loss == None: 
			self.loss = np.mean(np.square(errors))
		else: 
			self.loss = np.mean(np.array(np.mean(np.square(errors)),self.loss))
		#reversing to start from the end
		layers.reverse()
		self.weights.reverse()
		self.bias.reverse()
		#backpropagation
		for i in range(self.layers-1):
			# Calculating the gradient 
			gradients = self.disigmoid(layers[i])
			gradients = errors * gradients 
			gradients = gradients * self.learningRate
			# Calculating the deltas
			weights_deltas = np.outer(gradients, layers[i+1])
			# Updating the weights between the output and the hidden
			self.weights[i] = weights_deltas + self.weights[i]
			# Updating the bias of the output layer
			self.bias[i] = np.add(self.bias[i], gradients)

			weight_transposed = np.transpose(self.weights[i])
			errors = weight_transposed @ errors


		# Going back to the normal state.
		self.weights.reverse()
		self.bias.reverse()
		layers.reverse()
	
	def save(self):
		pass
