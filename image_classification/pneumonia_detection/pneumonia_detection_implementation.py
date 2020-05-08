import numpy as np
import os
import cv2
import random
import pickle

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

class pneumonia:

	def __init__(self):
		self.categories = ['normal', 'pneumonia']
		self.x = []
		self.y = []
		self.model = Sequential()

	# in this exact case, i have just two classes of data, but doing like this, i have an algorithm which can be used for 
	# more classes in the future
	def prepare_data(self, datadir, img_size=28):
		file_list = []
		class_list = []

		data = []
		img_size = 28

		for category in self.categories: 
			path = os.path.join(datadir, category)
			class_index = self.categories.index(category)
			for img in os.listdir(path):
				try:
					img_array = cv2.imread(os.path.join(path, img), cv2.imread_grayscale) # parse the image to an array in greyscale
					new_array = cv2.resize(img_array, (img_size, img_size)) # Resize the image to the correct size
					data.append([new_array, class_num]) # Add it to the array
				except:
					print(f'Error processing {img}')
					pass

		random.shuffle(data) # randomiza a ordem dos arquivos

		for features, label in data:
			self.x.append(features)
			self.y.append(label)

		self.x = np.array(x).reshape(-1, img_size, img_size, 1)
		return self


	def create_train_model(self):
		x = self.x
		y = self.y

		x /= 255

		# 3 convolutional layers
		self.model.add(Conv2D(32, (3, 3), input_shape = x.shape[1:]))
		self.model.add(Activation("relu"))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Conv2D(64, (3, 3)))
		self.model.add(Activation("relu"))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Conv2D(64, (3, 3)))
		self.model.add(Activation("relu"))
		self.model.add(MaxPooling2D(pool_size=(2,2)))
		self.model.add(Dropout(0.25))

		# 2 hidden layers
		self.model.add(Flatten())
		self.model.add(Dense(128))
		self.model.add(Activation("relu"))

		self.model.add(Dense(128))
		self.model.add(Activation("relu"))

		# The output layer with 13 neurons, for 13 classes
		self.model.add(Dense(13))
		self.model.add(Activation("softmax"))

		# Compile
		self.model.compile(loss="sparse_categorical_crossentropy",
				optimizer="adam",
				metrics=["accuracy"])

		self.model.fit(x, y, batch_size=32, epochs=40, validation_split=0.1)

	def save_model(self):



