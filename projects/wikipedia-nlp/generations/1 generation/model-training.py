import math
import numpy as np
import string
import json
import os
import sys
import logging
import boto3
import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from fastprogress.fastprogress import master_bar

from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

nltk.download('punkt')
nltk.download('stopwords')

chars = ''
maxlen = 60


def setup():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    load_dotenv('.env')
    global chars
    global maxlen


def shutdown(seconds=0, os='linux'):
    if os == 'linux':
        os.system(f'sudo shutdown -h -t sec {seconds}')
    elif os == 'windows':
        os.system(f'shutdown -s -t {seconds}')


def downloadDataset():
    s3 = boto3.client('s3')
    bucket = os.getenv('S3_DATASET_BUCKET')
    file = 'wikipedia-content-dataset.json'
    s3.download_file(bucket, file, file)
    logging.info(f'dataset downloaded')


def prepareData(dataFile):
    f = open(dataFile,)
    data = json.load(f)

    content = list(data[x] for x in data.keys())
    text = ''

    for c in content:
        for i in c:
            text += i

    logging.info(f'Corpus length: {len(text)}')

    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    text = ''
    for c in words:
        text += c
        text += ' '
    text = text.strip()

    logging.info(f'Finished to load file')
    return text


def prepareTrainingData(text):
    step = 3
    sentences = []
    next_chars = []

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    logging.info(f'Number of sequences: {len(sentences)}')

    chars = sorted(list(set(text)))
    logging.info(f'Unique characters: {len(chars)}')
    char_indices = dict((char, chars.index(char)) for char in chars)

    logging.info(f'Vectorizing text')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    logging.info(f'Finished to prepare data')
    return x, y


def prepareTrainModel(x, y):
    model = Sequential([
        LSTM(len(chars), return_sequences=True,
             input_shape=(maxlen, len(chars))),
        LSTM(len(chars), return_sequences=True),
        LSTM(len(chars)),
        Dense(len(chars), activation='relu'),
        Dropout(0.2),
        Dense(len(chars), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    logging.info(f'Starting to train model')
    mb = master_bar(range(1, 100))
    for epoch in mb:
        mb.comment = f'Epoch: {epoch}'
        model.fit(x, y, batch_size=128, epochs=1)

    logging.info(f'Finished to train model')
    return model


def saveModel(model):
    logging.info(f'Saving model to S3')
    s3 = boto3.client('s3')
    file = 'wikipedia-nlp.hdf5'
    gen = os.getenv('GENERATION')
    bucket = os.getenv('S3_BUCKET')

    model.save(file)

    s3.upload_file(file, bucket, gen+'/'+file)
    return 0


def main():
    setup()
    downloadDataset()
    text = prepareData('wikipedia-content-dataset.json')
    x, y = prepareTrainingData(text)
    model = prepareTrainModel(x, y)
    saveModel(model)
    logging.info(f'Model training finished and file saved to s3.')
    shutdown()


if __name__ == "__main__":
    main()
