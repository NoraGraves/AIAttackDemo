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
                 'Emma Stone':6, 'Greta Thunberg':7, 'Jackie Chan':8,
                 'Malala':9, 'Manuel Neuer':10, 'Mark Forster':11,
                 'Michael Jordan':12, 'Namika':13, 'Olaf Schulz':14,
                 'Olivia Rodrigo':15, 'Rihanna':16, 'Ryan Gosling':17,
                 'Sandra Oh':18, 'Serena Williams':19, 'Simu Lui':20,
                 'Zendaya':21}

# Given a .keras filepath, returns a keras object with those weights
def load_model_from_file(filepath):
    return keras.models.load_model(filepath)
    
# Given a preprocessed image and a keras model, returns the index the
# model predicts
def predict_index(image, model, verbose=None):
    probabilities = model.predict(image)
    return int(np.argmax(probabilities))

# Given a preprocessed image and a keras model, prints the name the
# model predicts for the given input, returns the index
def predict(image, model):
    index = predict_index(image, model)
    print(f'You look like {INDEX_2_NAME[index]}!')
    return int(index)

# Given a preprocessed image and a keras model, prints the probabilities
# for each celebrity
def print_all_probs(image, model):
    probabilities = model.predict(image, verbose=None)
    indices = np.argsort(probabilities)
    for i in indices:
        prob = probabilities[0][i]
        name = index2name(i)
        print(f'{name : <16} {(prob*100):.3f}%')

# returns the index of a name
def name2index(name):
    if name in NAME_2_INDEX:
        return int(NAME_2_INDEX[name])
    else:
        print(f'{name} is not in the dataset. Check the spelling and formatting, or try a different name.')

# returns the name of an index
def index2name(index):
    if index >= 0 & index <= 20:
        return str(INDEX_2_NAME[index])
    else:
        print(f'{index} is not a valid index.')
        return ''

# returns the filepath for a celeb image based on index
# designed for the specific Colab Notebook for this project!
def index2filepath(index):
    index_str = f'{index:02d}'
    folder_name = index_str + ''.join(index2name(index).split())
    filepath = f'/content/AIAttackDemo/sample_images/{folder_name}/{index_str}.jpeg'
    return filepath
