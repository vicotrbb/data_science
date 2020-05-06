# Neural network imports
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils

# Accuracy measure imports
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reconfigura o conjunto para um vetor de entrada de 28x28 pixels
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Padroniza os dados pra facilitar no treinamento
x_train /= 255
x_test /= 255

# encode utilizando o utilitario do numpy imbutido no keras
n_classes = 10
print(f'Formato antes do encoding: {y_train.shape}')
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
print(f'Formato depois do encoding: {y_train.shape}')

# Constroi a arquitetura da rede neural
model = Sequential()
# Camada convolucional
model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', 
                 activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(1, 1)))
model.add(Flatten())
# Camada oculta
model.add(Dense(100, activation='relu'))
# Camada de saida
model.add(Dense(10, activation='softmax'))

# Compila o modelo sequencial
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], 
              optimizer='adam')
# Treina o modelo
model.fit(x_train, y_train, batch_size=128, epochs=10, 
          validation_data=(x_test, y_test))

img = Image.open(r"mnist_test.png") 
img = img.resize((28, 28))
# convert rgb to grayscale
img = img.convert('L')
img = np.array(img)
# reshaping to support our model input and normalizing
img = img.reshape(1, 28, 28, 1)
img = img / 255.0
# predicting the class
res = model.predict([img])[0]
print(np.argmax(res))
print(max(res) * 100)