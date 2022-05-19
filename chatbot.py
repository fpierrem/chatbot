import json
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow import convert_to_tensor
from tensorflow.keras.models import load_model
import numpy as np
import random

THRESHOLD = 0.5
default_response = 'I don\'t understand, sorry.'

def load_files():
    vocabulary = pickle.load(open('vocabulary.pkl','rb'))
    labels = pickle.load(open('labels.pkl','rb'))
    intents_data = json.loads(open('intents.json','r').read())
    model = load_model('intent_recognizer')
    return vocabulary,labels,intents_data,model

def get_input():
    sentence = input()
    return sentence

def transform_input(sentence, vocabulary):
    # Transform input sentence into bag of words and then tensor
    sentence_words = word_tokenize(sentence.lower())
    wnl = WordNetLemmatizer()
    sentence_lemmas = [wnl.lemmatize(word) for word in sentence_words]

    bow = [0] * len(vocabulary)
    for index,word in enumerate(vocabulary):
        if word in sentence_lemmas:
            bow[index] = 1
    bow = np.array(bow)
    bow = np.expand_dims(bow,0)
    t = convert_to_tensor(bow)
    return t

def predict_label(sentence_vector,model,labels):
    prediction = model(sentence_vector)
    predicted_label = labels[np.argmax(prediction)] if np.amax(prediction) > THRESHOLD else None
    return predicted_label

def respond(intents_data,labels,predicted_label):
    # Pick a random response
    if not predicted_label:
        return default_response
    responses = next(intent for intent in intents_data['intents'] if intent['tag'] == predicted_label)['responses']
    response = random.choice(responses)
    return response

def __main__():
    vocabulary,labels,intents_data,model = load_files()
    print('Write something:\n')
    while True:
        sentence = get_input()
        sentence_vector = transform_input(sentence, vocabulary)
        predicted_label = predict_label(sentence_vector,model,labels)
        response = respond(intents_data,labels,predicted_label)
        print(response)

if __name__ == '__main__':
    __main__()
