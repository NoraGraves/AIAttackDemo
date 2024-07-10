import keras
import numpy as np

# Stores the names corresponding to the indices in the model
__INDEX_TO_NAME = ['Alica Schmidt', 'Angela Merkel', 'Barack Obama',
                 'Bruno Mars', 'Dwayne Johnson', 'Ed Sheeran',
                 'Emma Stone', 'Greta Thunberg', 'Jackie Chan',
                 'Malala', 'Manuel Neuer', 'Mark Forster',
                 'Michael Jordan', 'Namika', 'Olaf Schulz',
                 'Olivia Rodrigo', 'Rihanna', 'Ryan Gosling',
                 'Sandra Oh', 'Serena Williams', 'Simu Lui', 'Zendaya']

# Given a .keras filepath, returns a keras object with those weights
def load_model_from_file(filepath):
    return keras.models.load_model(filepath)
    
# Given a preprocessed image and a keras model, returns the index the
# model predicts
def predict_index(image, model, verbose=0):
    probabilities = model.predict(image)
    return int(np.argmax(probabilities))

# Given a preprocessed image and a keras model, prints the name the
# model predicts for the given input, returns the index
def predict(image, model):
    index = predict_index(image, model)
    print(f'You look like {__INDEX_TO_NAME[index]}!')
    return int(index)

# Given a preprocessed image and a keras model, prints the probabilities
# for each celebrity
def print_all_probs(image, model):
    probabilities = model.predict(image, verbose=0)
    indices = np.argsort(probabilities)
    indices = np.flip(indices[0])
    for i in indices:
        prob = probabilities[0][i]
        name = index2name(i)
        print(f'{name : <16} {(prob*100):.3f}%')

# returns the index of a name
def name2index(name):
    if name in __INDEX_TO_NAME:
        return __INDEX_TO_NAME.index(name)
    else:
        raise ValueError(f'{name} is not in the dataset. Check the spelling and formatting, or try a different name.')

# returns the name of an index
def index2name(index):
    if (index >= 0) and (index <= 21):
        return str(__INDEX_TO_NAME[index])
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
