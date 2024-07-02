import keras
import numpy as np

# Stores the names corresponding to the indices in the model
INDEX_2_NAME = ['Alica Schmidt', 'Angela Merkel', 'Barack Obama',
                 'Bruno Mars', 'Dwayne Johnson', 'Ed Sheeran',
                 'Emma Stone', 'Greta Thunberg', 'Jackie Chan',
                 'Malala', 'Manuel Neuer', 'Mark Forster',
                 'Michael Jordan', 'Namika', 'Olaf Schulz',
                 'Olivia Rodrigo', 'Rihanna', 'Ryan Gosling',
                 'Sandra Oh', 'Serena Williams', 'Simu Lui', 'Zendaya']
NAME_2_INDEX = {'Alica Schmidt':0, 'Angela Merkel' :1, 'Barack Obama':2,
                 'Bruno Mars':3, 'Dwayne Johnson':4, 'Ed Sheeran':5,
                 'Emma Stone':6, 'Greta Thunberg':7, 'Jackie Chan':7,
                 'Malala':8, 'Manuel Neuer':9, 'Mark Forster':10,
                 'Michael Jordan':11, 'Namika':12, 'Olaf Schulz':13,
                 'Olivia Rodrigo':14, 'Rihanna':15, 'Ryan Gosling':16,
                 'Sandra Oh':17, 'Serena Williams':18, 'Simu Lui':19,
                 'Zendaya':20}

# Given a .keras filepath, returns a keras object with those weights
def load_model_from_file(filepath):
    return keras.models.load_model(filepath)

# Given a preprocessed image and a keras model, prints the name the
# model predicts for the given input
def predict(image, model):
    probabilities = model.predict(image)
    index = np.argmax(probabilities)
    print(f"You look like {INDEX_2_NAME[index]}!")

# Given a preprocessed image and a keras model, prints the probabilities
# for each celebrity
def print_all_probs(image, model):
    probabilities = model.predict(image)
    for i, prob in enumerate(probabilities[0]):
        print(f'{INDEX_2_NAME[i] : <16} {(prob*100):.3f}%')


# Given a preprocessed image and a keras model, returns the index the
# model predicts
def predict_index(image, model):
    probabilities = model.predict(image)
    return int(np.argmax(probabilities))

# returns the index of a name
def name2index(name):
    if name in NAME_2_INDEX:
        return NAME_2_INDEX[name]
    else:
        print(f'{name} is not in the dataset. Check the spelling and formatting, or try a different name.')

# returns the name of an index
def index2name(index):
    if index >= 0 & index <= 20:
        return INDEX_2_NAME[index]
    else:
        print(f'{index} is not a valid index.')
