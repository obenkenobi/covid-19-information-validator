import os

import tensorflow as tf
from tensorflow import keras
import json, csv
import numpy as np
from process import preprocess_txt, listify_txt, word_indexer, split_covid_data_entry, split_covid_data

def preprocess_covid19_data(collection, max_txt_size=255, splice=True):
    """
    Takes in a list of dictionaries where each dictionary takes the form:
    {
        'text': a string block,
        'valid': an integer
    }
    and max_txt_size which will set the size of your texts in collection to that size.
    It will return a list of encoded texts, a list of valid labels, and a word_index that maps a word to an encoded integer form
    """
    if splice:
        collection = split_covid_data(collection)
    np.random.shuffle(collection)
    labels = np.array([data['valid'] for data in collection], dtype=np.int32)
    texts =  [data['text'] for data in collection]
    word_dump = []
    for text in texts:
        word_dump.extend(listify_txt(text))
    word_index = word_indexer(word_dump)
    del word_dump
    encoded_txt = np.array([preprocess_txt(text, word_index, max_txt_size) for text in texts])
    return encoded_txt, labels, word_index

def train_info_validator(x_train, y_train, embeding_dim=(88000,16), epochs=7, batch_size=None, validation_data=None):
    """
    creates and trains neural network with processed training data
    """
    model = keras.Sequential()
    model.add(keras.layers.Embedding(embeding_dim[0],embeding_dim[1]))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(64,activation="relu"))
    model.add(keras.layers.Dense(16,activation="relu"))
    model.add(keras.layers.Dense(3,activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=1)
    return model

def train_save_info_validator(x_train, y_train, embeding_dim=(88000,16), epochs=7, batch_size=None, validation_data=None):
    """
    trains neural network with training data and saves it
    """
    model = train_info_validator(x_train, y_train, embeding_dim=embeding_dim, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
    if validation_data == None:
        print('no validation data added')
    else:
        accuracy = model.evaluate(validation_data[0], validation_data[1])[1]
        print(f'Finished training with model with validation accuracy {accuracy}')
        print(f'Will now save model')
    file_dir = os.path.dirname(os.path.abspath(__file__))
    model.save(os.path.join(file_dir, "covid19_info_validator.h5"), include_optimizer=False)
    return model

def train_neuralnetwork(data_filepath=None, word_index_filename=None, splice=True):
    """
    trains neural network from csv data
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    if data_filepath == None:        
        data_filepath = os.path.join(file_dir, 'data', 'text_data.csv')

    if word_index_filename == None:
        word_index_filename = os.path.join(file_dir, 'data', 'word_decode.json')

    with open(data_filepath) as f: 
        training_collection = list(csv.DictReader(f))

    train_size = len(training_collection) - len(training_collection)//5

    x_train, y_train, word_index = preprocess_covid19_data(training_collection, splice=splice) # preprocess training collection
    x_train, y_train, x_val, y_val = x_train[:train_size], y_train[:train_size], x_train[train_size:], y_train[train_size:]

    with open(word_index_filename, 'w') as f:
        json.dump(word_index, f) # store word index

    model = train_save_info_validator(x_train, y_train, embeding_dim=(len(word_index), 16), epochs=23, validation_data=(x_val, y_val))
    
    # clear resources
    del model
    tf.keras.backend.clear_session()

def validate_txt_with_index(txt, word_index, model=None):
    """
    Takes in text and an encoding word index and returns an integer determining if a text provides valid information 
    or misinformation about the COVID-19 virus.
    If a text is valid, then the function returns 0.
    If a text is neither valid nor misinformation, returns 1.
    If a text is misinformation, returns 2
    You can also optionally pass a model which represents a keras neural network instead of the function loading
    a pre-existing neural network
    """
    if not model:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        model = keras.models.load_model(os.path.join(file_dir, "covid19_info_validator.h5"), compile=False)
        
    encoded = preprocess_txt(txt, word_index=word_index)
    prediction = model.predict(np.array([encoded], dtype=np.int32))[0] # a numoy of int33 datatype is only permitted
    
    # clear resources
    del model
    tf.keras.backend.clear_session()
    
    return np.argmax(prediction)    

def validate_txt(txt, word_index_filepath=None, model=None):
    """
    Takes in text and the filename of word index json file and returns an integer determining if a text provides valid information 
    or misinformation about the COVID-19 virus.
    If a text is valid, then the function returns 0.
    If a text is neither valid nor misinformation, returns 1.
    If a text is misinformation, returns 2
    You can also optionally pass a model which represents a keras neural network instead of the function loading
    a pre-existing neural network
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    if word_index_filepath==None:
        word_index_filepath = os.path.join(file_dir, 'data' + os.sep + 'word_decode.json')
    with open(word_index_filepath) as f:
        word_index = json.load(f)
    return validate_txt_with_index(txt, word_index, model=model)