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

	# in this exact case, i have just two classes of data, but doing like this, i have an algorithm which can be used for 
	# more classes in the future
def prepare_data(datadir, img_size=28):
	file_list = []
	class_list = []
	x = []
	y = []
	data = []
	error = False
	categories = ['normal', 'pneumonia']

	for category in categories: 
		path = os.path.join(datadir, category)
		class_index = categories.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # parse the image to an array in greyscale
				new_array = cv2.resize(img_array, (img_size, img_size)) # Resize the image to the correct size
				data.append([new_array, class_index]) # Add it to the array
			except Exception as e:
				error = True
				pass

	random.shuffle(data) # randomiza a ordem dos arquivos

	for features, label in data:
		x.append(features)
		y.append(label)

	x = np.array(x).reshape(-1, img_size, img_size, 1)
	if error:
		print('Erro ao processar algums imagens')
	else:
		print('Imagens processadas com sucesso')
	return x, y


class Pneumonia:

	def __init__(self, train_dir, test_dir):
		self.model = Sequential()
		self.img_size = 28
		self.categories = ['normal', 'pneumonia']
		self.x_train, self.y_train = prepare_data(train_dir)
		self.x_test, self.y_test = prepare_data(test_dir)


	def create_train_model(self):
		self.model = Sequential()
		self.x_train /= 255

		# convolutional layers
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

		# hidden layers
		self.model.add(Flatten())
		self.model.add(Dense(128))
		self.model.add(Activation("relu"))

		self.model.add(Dense(128))
		self.model.add(Activation("relu"))

		# output layer
		self.model.add(Dense(2))
		self.model.add(Activation("softmax"))

		# Compile
		self.model.compile(loss="sparse_categorical_crossentropy",
				optimizer="adam",
				metrics=["accuracy"])

		self.model.fit(self.x_train, self.y_test, batch_size=32, epochs=40, validation_split=0.1,
			verbose=1, validation_data=(self.x_test, self.y_test))

		return self


	def predict_image(self, file):
		img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
		new_array = cv2.resize(img_array, (self.img_size, self.img_size))
		img = new_array.reshape(-1, self.img_size, self.img_size, 1)
		prediction = model.predict([img])
		prediction = list(prediction[0])
		print(self.categories[prediction.index(max(prediction))])


#Pneumonia('pneumonia_dataset/train', 'pneumonia_dataset/test')




