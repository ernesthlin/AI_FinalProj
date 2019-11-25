from preprocessing import PreProcessor
import math

class Perceptron():

	def __init__(self, y, weights, digit_weights):
		self.y_values = y
		self.weights = weights
		self.digit_weights = digit_weights
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

##########################################################################################################
											'''
										Face Functions  		
											'''
##########################################################################################################
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
			if (self.y_values[i] == 0) and (guesses[i] <= 0): 
				correct += 1
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
		flattened = flatten_list(self, X_train)
		self.weights = [1 for i in range(pixels+1)]

		# now itterate through the flattened list, and do the training on each image
		flattened_index = 0
		guesses = []
		for i in range(len(X_train)):
			current_image = []
			for j in range(pixels): 
				current_image.append(flattened[flattened_index])
				flattened_index += 1			
			# now, we have current image, so calculate the f value
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
					self.weights = update_weights_face(self, current_image, add) 
					guess = calculate_f_face(self, current_image)
			guesses[i] = guess
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

##########################################################################################################
											'''
										Digit Functions  		
											'''
##########################################################################################################
'''
	Calculate f values for each digit 0-9 and take the max value
		f(xi, weights) = w0*o(xi) + w1*o(xi) + w2*o(xi)... + w4071*o(xi) + w4072
	Self.weights is a 2D array of size 10 with indeces 0-9 representing those digits, with thier cooresponding list 
	equal to the weights for that digit 
	Returns the index of the max f value, or the 'digit' that we have guessed with the highest certainty 
'''
def calculate_f_digit(self, xi): 
	f_digits = []
	for i in range(len(f_digits)):
		sum = 0
		for j in range(len(xi)):
			sum += (self.digit_weights[i][j] * xi[j])
		sum += self.digit_weights[i][-1]
		f_digits[i] = sum
	return f_digits.index(max(f_digits))

'''
	If correct guess, move on. 
	If incorrect guess, 2 things must happen: 
		1. The incorrect guess's weights must be either decremented by each of its features
		2. The weights for the correct digit's weights must be incremented, the opposite of step 1 
'''
def check_guess_accuracy_and_update(self, guess, index, xi): 
	if self.y_values[index] != guess: 
		# decrement incorrect 
		self.digit_weights[guess] = [self.digit_weights[guess][i] - xi[i] for i in range(len(self.digit_weights[guess]))-1]
		self.digit_weights[guess][-1] = 1 - self.digit_weights[guess][-1]

		# increment correct
		self.digit_weights[index] = [self.digit_weights[index][i] + xi[i] for i in range(len(self.digit_weights[index]))-1]
		self.digit_weights[index][-1] += 1
		return False
	return True

def percentage_correct_digit(self, guesses): 
	correct_zipped = list(zip(self.y_values, guesses))
	correct = [i for i, j in correct_zipped if i == j]
	return (len(correct)/len(self.y_values)) * 100

'''
	weights will be of size 4720 + 1 (the w0 is the extra weight)
	Length of training set (X_train is 451 (usually) aka how many images we have)
		- our features are each individual pixel, and each pixel has either a 0, 1, or 2 (for numbers) 
		- f(xi, weights) = w0 + w1*o(xi) + w2*o(xi) + w3*o(xi)... + w4072*o(xi)
'''
def train_digit(self, X_train): 
	pixels = 28 * 29 # digit image pixels
	# get flattened list
	flattened = flatten_list(self, X_train)
	# initialize the weights list, to be 1 more than the # pixels in the image
	for i in range(10)
		self.digit_weights[i] = [1 for j in range(pixels+1)]

	flattened_index = 0
	guesses = []
	for i in range(len(X_train)): 
		current_image =[]
		for j in range(pixels): 
			current_image.append(flattened[flattened_index])
			flattened_index += 1			
		# Calculate the f value and assign the 'digit' we have guessed to guess
		guess = calculate_f_digit(self, current_image)
		check = check_guess_accuracy_and_update(self, guess, i, current_image)
		while (check is False): 
			guess = calculate_f_digit(self, current_image)
			check = check_guess_accuracy_and_update(self, guess, i, current_image)
		guesses[i] = guess
	percent_correct = percentage_correct_digit(self, guesses)
	return percent_correct



		