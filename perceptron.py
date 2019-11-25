from preprocessing import PreProcessor
import math

class Perceptron():

	def __init__(self, y, weights):
		self.y_values = y
		self.weights = weights
	'''
		flatten the image into a single list of size 4072
		X_train is a list of [([0,1,2..],y),([1,0,2...],y1),([1,2,0...],y2)..]
	'''
	def flatten_list(self, X_train,): 
		# convert list of training data into a flattened list, extract the y values
		x_list = [i for i, j in X_train]
		self.y_values = [j for i, j in X_train]
		# get every image, flatten from list of [[]] to list of [], to list of ints
		list_of_images = [j for i in x_list for j in i]
		flattened = [j for i in list_of_images for j in i]
		return flattened

	'''
		calculates 
		- f(xi, weights) = w0*o(xi) + w1*o(xi) + w2*o(xi)... + w4071*o(xi) + w4072
	'''
	def calculate_f_face(self, xi): 
		f_xi_weights = 0
		for i in range(len(xi)):
			f_xi_weights += (self.weights[i] * xi[i])
		# add our last feature to the end, the newest one 
		f_xi_weights += xi[-1]
		return f_xi_weights

	'''
		update the wieghts when a prediction is incorrect
	'''
	def update_weights_face(self, xi, add): 
		# have to subtract
		updated = []
		if add is False: 
			for i in range(len(xi)):
				updated[i] = self.weights[i] - x[i] 
			updated[len(xi)] = 1 - self.weights[-1]
		# have to add
		elif add is True: 
			for i in range(len(xi)):
				updated[i] = self.weights[i] + x[i] 
			updated[len(xi)] = 1 + self.weights[-1] 
		return updated

	'''
		compares values of the final final guesses with the actual ys
		0 -> <= 0 is correct (is not an image) 
		1 -> >= 0 is correct (is an image) 
	'''
	def percentage_correct_face(self, guesses): 
		correct = 0
		total = len(self.y_values)
		for i in range(len(self.y_values)): 
			# correct
			if (self.y_values[i] == 0) and (guesses[i] <= 0): 
				correct += 1
			# correct
			elif (self.y_values[i] == 1) and (guesses[i] >= 0):
				correct += 1

		return (correct/total) * 100


	'''
		if_face is a 0 for false and a 1 for true
			- if 0 then do digits if false then do faces
		weights will be of size 4720 + 1 (the w0 is the extra weight)
		- our features are each individual pixel, and each pixel has either a 
		0, 1, or 2 (for numbers) and a 0, 1 for faces
		- f(xi, weights) = w0 + w1*o(xi) + w2*o(xi) + w3*o(xi)... + w4072*o(xi)
	'''
	def train_face(self, X_train): 
		pixels = 70 * 61 # face image pixels

		# get flattened list
		flattened = flatten_list(self, X_train)
		# initialize the weights list, to be 1 more than the # pixels in the image
		self.weights = [1 for i in range(pixels+1)]

		# now itterate through the flattened list, and do the training on each image
		flattened_index = 0
		guesses = []
		for i in range(len(X_train)):
			current_image = []
			for j in range(pixels): 
				current_image.append(flattened[flattened_index])
				flattened_index += 1			# now, we have current image, so calculate weights
			guess = calculate_f_face(self, current_image)
			
			# if not face and we are positive, have to subtract 
			if (self.y_values[i] == 0) and (guess >= 0): 
				add = False
				while (guess >= 0): 
					# update the weights
					self.weights = update_weights_face(self, current_image, add) 
					# then, recalculate
					guess = calculate_f_face(self, current_image)
			
			# if face and we are negative, have to subtract 
			elif (self.y_values[i] == 1) and (guess <= 0): 
				add = True
				while (guess <= 0): 
					# update the weights
					self.weights = update_weights_face(self, current_image, add) 
					# then, recalculate
					guess = calculate_f_face(self, current_image)
			guesses[i] = guess
		# now calculate percentage correct
		percent_correct = percentage_correct_face(self, guesses)
		return percent_correct
			


	def predict_face(self, x_input): 
		# assign x and y values, flatten the list of lists
		x, y = x_input
		flattened = [j for i in x for j in i]

		guess = 0
		for i in range(len(flattened)): 
			guess += (self.weights[i] * flattened[i])
		guess += self.weights(len(flattened))

		# if it should NOT be a face
		if y == 0: 
			if guess <= 0: 
				return true

		# if it SHOULD be a face
		elif y == 1: 
			if guess >= 0: 
				return true

		return false

		