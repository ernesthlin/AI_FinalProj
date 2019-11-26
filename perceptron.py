from preprocessing import PreProcessor
import math

class Perceptron():

	def __init__(self, y, weights, digit_weights):
		self.y_values = y
		self.weights = weights
		self.digit_weights = digit_weights

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
		pixels = 70 * 61 # face image pixels
		self.weights = [0 for i in range(pixels+1)]
		self.y_values = y_train
		# now itterate through the flattened list, and do the training on each image
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
		#print("length of the list at : " + str(i) + " " + str(self.digit_weights[i][-1]))
		f_digits = []
		for i in range(10):
			#print("i = " + str(i))
			#print("length of the list at : " + str(i) + " " + str(len(self.digit_weights[i])))
			sum = 0
			for j in range(len(xi)):
				#print("j = " + str(j) + " xi[j]: " + str(xi[j]))
				sum += (self.digit_weights[i][j] * xi[j])
			sum += self.digit_weights[i][-1]
			#print("sum = " + str(sum))
			f_digits.append(sum)
		maxInt = f_digits.index(max(f_digits))
		#print("the f is " + str(maxInt))
		#print(self.digit_weights)
		return maxInt


	'''
		If correct guess, move on. 
		If incorrect guess, 2 things must happen: 
			1. The incorrect guess's weights must be either decremented by each of its features
			2. The weights for the correct digit's weights must be incremented, the opposite of step 1 
	'''
	def check_guess_accuracy_and_update(self, guess, y, xi): 
		multiplier = 1 # 1 default, % with .9
		print(" guess is " + str(guess) + " y is " + str(y))
		#print(" guess is " + str(guess))
		
		self.digit_weights[guess].clear()
		self.digit_weights[y].clear()

		add_to_guesses = []
		add_to_ys = []
		
		print(self.digit_weights[guess][0])

		if y == guess: 
			return True
		else: 
			for j in range(len(xi)):
				print("j is : " + str(j))
				print(self.digit_weights[guess][0])
				#print("the xi is " + str(xi[j]) + " the weight guess is " + str(self.digit_weights[guess][j]))
				add_to_guesses.append(self.digit_weights[guess][j] - (multiplier * xi[j]))
				add_to_ys.append(self.digit_weights[y][j] + (multiplier * xi[j]))
			add_to_guesses.append(self.digit_weights[guess][-1] - (multiplier*1))
			add_to_ys.append(self.digit_weights[y][-1] + (multiplier*1))

			self.digit_weights[guess] = add_to_guesses
			self.digit_weights[y] = add_to_ys
			return False


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
	def train_digit(self, X_train, y_train): 
		pixels = 28 * 29 # digit image pixels
		self.digit_weights = [[0] * (pixels+1)] * 10
		self.y_values = y_train
		# initialize the weights list, to be 1 more than the # pixels in the image
		guess = 0
		guesses = []
		for i in range(len(X_train)): 
			current_image = X_train[i]			
			# Calculate the f value and assign the 'digit' we have guessed to guess
			guess = self.calculate_f_digit(current_image)
			guesses.append(guess)
			print(self.digit_weights[guess][0])
			check = self.check_guess_accuracy_and_update(guess, self.y_values[i], current_image)
			print("did first check")
			# while (check is False): 
			# 	guess = self.calculate_f_digit(current_image)
			# 	check = self.check_guess_accuracy_and_update(guess, self.y_values[i], current_image)
		percent_correct = self.percentage_correct_digit(guesses)
		return percent_correct


	def predict_digit(self, X_Validation, y_Validation): 
		self.y_values = y_Validation
		guess = 0
		guesses = []
		for i in range(len(X_Validation)):
			current_image = X_Validation[i]
			# now, we have current image, so calculate the f value
			guess = self.calculate_f_digit(current_image)
			guesses.append(guess)

		percent_correct = self.percentage_correct_digit(guesses)
		return percent_correct

# for testing purposes
def main():
	# face_data = PreProcessor("/Users/nicolascarchio/Desktop/AI_FinalProj/data/facedata/facedatatrain", "/Users/nicolascarchio/Desktop/AI_FinalProj/data/facedata/facedatatrainlabels")
	# # create structure
	# perceptron = Perceptron([],[],[]) 
	# correct = perceptron.train_face(face_data.X, face_data.y)
	# print("The percentage correct on training data is " + str(correct))


	# face_data_validation = PreProcessor("/Users/nicolascarchio/Desktop/AI_FinalProj/data/facedata/facedatavalidation", "/Users/nicolascarchio/Desktop/AI_FinalProj/data/facedata/facedatavalidationlabels")
	# # create structure
	# correctV = perceptron.predict_face(face_data_validation.X, face_data_validation.y)
	# print("The percentage correct on validation data is " + str(correctV))


	digit_data = PreProcessor("/Users/nicolascarchio/Desktop/AI_FinalProj/data/digitdata/trainingimages", "/Users/nicolascarchio/Desktop/AI_FinalProj/data/digitdata/traininglabels")
	# create structure
	perceptron_digit = Perceptron([],[],[]) 
	correctd = perceptron_digit.train_digit(digit_data.X, digit_data.y)
	print("The percentage correct on training data is " + str(correctd))


	digit_data = PreProcessor("/Users/nicolascarchio/Desktop/AI_FinalProj/data/digitdata/validationimages", "/Users/nicolascarchio/Desktop/AI_FinalProj/data/digitdata/validationlabels")
	# create structure
	correctVd = perceptron_digit.predict_digit(digit_data.X, digit_data.y)
	print("The percentage correct on validation data is " + str(correctVd))




if __name__ == "__main__":
	main()	







			