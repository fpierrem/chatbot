# Train intent recognizer

import random
import json
import pickle
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from download_nltk_modules import download_modules

import tensorflow as tf

# Download required NLTK module
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    download_modules('punkt')

# Build vocabulary, list of labels, list of documents (ie (word_patterns, tag) tuples)
vocabulary = []
labels = []
documents = []
ignore_letters = ['?','!','.','\'s']

intents = json.loads(open('intents.json','r').read())

for intent in intents['intents']:
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
    for sentence in intent['sentences']:
        words = word_tokenize(sentence.lower())
        for word in words:
            if word not in vocabulary:
                vocabulary.append(word)
        document = (intent['tag'],words)
        documents.append(document)

wnl = WordNetLemmatizer()
vocabulary = [wnl.lemmatize(word) for word in vocabulary if word not in ignore_letters]

pickle.dump(vocabulary,open('vocabulary.pkl','wb'))
pickle.dump(labels,open('labels.pkl','wb'))

# Transform documents into BOW vectors and labels into label vectors.
# x_train will be made of a list of document_vectors and y_train will be a list of tags
training_set = []

for document in documents:
    bow = [0] * len(vocabulary)
    sentences = [wnl.lemmatize(word) for word in document[1]]
    for index,word in enumerate(vocabulary):
        if word in sentences:
            bow[index] = 1
    label_vector = [0] * len(labels)
    index = labels.index(document[0])
    label_vector[index] = 1
    training_set.append([bow, label_vector])

random.shuffle(training_set)
training_set = np.array(training_set)
x_train = list(training_set[:,0])
y_train = list(training_set[:,1])

# Build, compile and train the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu',input_shape=(len(x_train[0]),)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(y_train[0]), activation='softmax')    
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=200)

model.save('intent_recognizer')