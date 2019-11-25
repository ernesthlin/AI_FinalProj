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
	# def flatten_list(self, X_train): 
	# 	# convert list of training data into a flattened list, extract the y values
	# 	x_list = [i for i, j in X_train]
	# 	self.y_values = [j for i, j in X_train]
	# 	# get every image, flatten from list of [[]] to list of [], to list of ints
	# 	list_of_images = [j for i in x_list for j in i]
	# 	flattened = [j for i in list_of_images for j in i]
	# 	return flattened

##########################################################################################################
										
										#Face Functions  		
										
##########################################################################################################
	'''
		calculates 
		- f(xi, weights) = w0*o(xi) + w1*o(xi) + w2*o(xi)... + w4071*o(xi) + w4072
	'''
	def calculate_f_face(self, xi): 
		f_xi_weights = 0
		# print("Xi")
		# print("weights = " + str(self.weights))
		for i in range(len(xi)):
			f_xi_weights += (self.weights[i] * xi[i])
		# add our last feature to the end, the newest one 
		f_xi_weights += xi[-1]
		#print("f is now: " + str(f_xi_weights))
		return f_xi_weights

	'''
		update the wieghts when a prediction is incorrect
	'''
	def update_weights_face(self, xi, add): 
		multiplier = .9 # 1 default, 73.6% - .9
		# have to subtract when add is false
		updated = []
		if add is False: 
			for i in range(len(xi)):
				updated.append(self.weights[i] - (multiplier)*xi[i])
			updated.append((multiplier)*1 - self.weights[-1])
		# have to add when add is true
		elif add is True: 
			for i in range(len(xi)):
				updated.append(self.weights[i] + (multiplier)*xi[i])
			updated.append((multiplier)*1 + self.weights[-1]) 
		#print("updated is " + str(updated))
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
		print(correct)
		print(total)
		return (correct/total) * 100

	'''
		if_face is a 0 for false and a 1 for true
			- if 0 then do digits if false then do faces
		weights will be of size 4720 + 1 (the w0 is the extra weight)
		- our features are each individual pixel, and each pixel has either a 
		0, 1, or 2 (for numbers) and a 0, 1 for faces
		- f(xi, weights) = w0 + w1*o(xi) + w2*o(xi) + w3*o(xi)... + w4072*o(xi)
	'''
	def train_face(self, X_train, y_train): 
		count = 0
		pixels = 70 * 61 # face image pixels
		self.weights = [0 for i in range(pixels+1)]
		self.y_values = y_train

		# now itterate through the flattened list, and do the training on each image
		flattened_index = 0
		guess = 0
		guesses = []
		for i in range(len(X_train)):
			current_image = X_train[i]
			# now, we have current image, so calculate the f value
			guess = self.calculate_f_face(current_image)
			guesses.append(guess)

			# if not face and we are positive, have to subtract 
			if (self.y_values[i] == 0) and (guess >= 0): 
				add = False
				while (guess >= 0): 
					# update the weights
					self.weights = self.update_weights_face(current_image, add) 
					# then, recalculate
					guess = self.calculate_f_face(current_image)
			
			# if face and we are negative, have to subtract 
			elif (self.y_values[i] == 1) and (guess <= 0): 
				add = True
				while (guess <= 0): 
					self.weights = self.update_weights_face(current_image, add) 
					guess = self.calculate_f_face(current_image)
			
		percent_correct = self.percentage_correct_face(guesses)
		return percent_correct
			

	def predict_face(self, X_Validation, y_Validation): 
		self.y_values = y_Validation
		guess = 0
		guesses = []
		for i in range(len(X_Validation)):
			current_image = X_Validation[i]
			# now, we have current image, so calculate the f value
			guess = self.calculate_f_face(current_image)
			guesses.append(guess)

		percent_correct = self.percentage_correct_face(guesses)
		return percent_correct

		# # assign x and y values, flatten the list of lists
		# x, y = x_input
		# flattened = [j for i in x for j in i]

		# guess = 0
		# for i in range(len(flattened)): 
		# 	guess += (self.weights[i] * flattened[i])
		# guess += self.weights(len(flattened))

		# # if it should NOT be a face
		# if y == 0: 
		# 	if guess <= 0: 
		# 		return 1
		# # if it SHOULD be a face
		# elif y == 1: 
		# 	if guess >= 0: 
		# 		return 1
		# return 0

##########################################################################################################
											
										# Digit Functions  		
											
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
	def train_digit(self, preProc): 
		pixels = 28 * 29 # digit image pixels
		# get flattened list
		#flattened = self.flatten_list(X_train)
		flattened = preProc.X
		# initialize the weights list, to be 1 more than the # pixels in the image
		for i in range(10): 
			self.digit_weights[i] = [0 for j in range(pixels+1)]

		flattened_index = 0
		guesses = []
		for i in range(len(X_train)): 
			current_image =[]
			for j in range(pixels): 
				current_image.append(flattened[flattened_index])
				flattened_index += 1			
			# Calculate the f value and assign the 'digit' we have guessed to guess
			guess = self.calculate_f_digit(current_image)
			check = self.check_guess_accuracy_and_update(guess, i, current_image)
			while (check is False): 
				guess = self.calculate_f_digit(current_image)
				check = self.check_guess_accuracy_and_update(guess, i, current_image)
			guesses[i] = guess
		percent_correct = self.percentage_correct_digit(guesses)
		return percent_correct

	def predict_digit(self, x_input): 
		# assign x and y values, flatten the list of lists
		x, y = x_input
		flattened = [j for i in x for j in i]

		digit = self.calculate_f_digit(flattened)

		if digit == y: 
			return true
		else: 
			return false

# for testing purposes
def main():
	face_data = PreProcessor("/Users/nicolascarchio/Desktop/AI_FinalProj/data/facedata/facedatatrain", "/Users/nicolascarchio/Desktop/AI_FinalProj/data/facedata/facedatatrainlabels")
	# create structure
	perceptron = Perceptron([],[],[]) 
	correct = perceptron.train_face(face_data.X, face_data.y)
	print("The percentage correct on training data is " + str(correct))


	face_data_validation = PreProcessor("/Users/nicolascarchio/Desktop/AI_FinalProj/data/facedata/facedatavalidation", "/Users/nicolascarchio/Desktop/AI_FinalProj/data/facedata/facedatavalidationlabels")
	# create structure
	correctV = perceptron.predict_face(face_data_validation.X, face_data_validation.y)
	print("The percentage correct on validation data is " + str(correctV))




if __name__ == "__main__":
	main()	







			