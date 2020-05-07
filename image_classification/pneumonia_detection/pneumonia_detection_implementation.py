import numpy as np
import os
import cv2
import random
import pickle

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt

# in this exact case, i have just two classes of data, but doing like this, i have an algorithm which can be used for more classes
# in the future
def prepare_data(datadir, img_size=28):
	file_list = []
	class_list = []

	train_dir_name = 'train'
	test_dir_name = 'test'

	categories = ['normal', 'pneumonia']
	data = []
	for category in categories: 
		path = os.path.join(datadir, category)
		class_index = categories.index(category)
		for img in os.listdir(path):
		try :
			img_array = cv2.imread(os.path.join(path, img), cv2.imread_grayscale) # parse the image to an array in greyscale
			new_array = cv2.resize(img_array, (img_size, img_size)) # Resize the image to the correct size
			data.append([new_array, class_num]) # Add it to the array
		except exception as e:
			print(f'Error processing {img}')
			pass

	random.shuffle(data) # randomiza a ordem dos arquivos
	x = []
	y = []

	X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

	# Creating the files containing all the information about your model
	pickle_out = open("X.pickle", "wb")
	pickle.dump(X, pickle_out)
	pickle_out.close()

	pickle_out = open("y.pickle", "wb")
	pickle.dump(y, pickle_out)
	pickle_out.close()
