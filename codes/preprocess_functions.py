from tqdm import tqdm
import re
from textblob import Word
import numpy as np

# Split at capitalize words and add underscore
def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def clean_trope(column):
    # Replace Longrunner with LongRunner
    column = column.str.replace('Longrunner', 'LongRunner', regex=False)
    # dataframe.loc[dataframe.Trope=='Longrunner', 'Trope'] = 'LongRunner'
    # Split at capitalize words and add underscore 
    column = column.apply(lambda w: convert(w))
    # Convert plural to singular to get rid of duplicates
    column = column.apply(lambda w: Word(w).singularize())
    # dataframe = dataframe.reset_index(drop=True)
    return column

# Decontract text
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", str(phrase))
    phrase = re.sub(r"can\'t", "can not", str(phrase))

    # general
    phrase = re.sub(r"n\'t", " not", str(phrase))
    phrase = re.sub(r"\'re", " are", str(phrase))
    phrase = re.sub(r"\'s", " is", str(phrase))
    phrase = re.sub(r"\'d", " would", str(phrase))
    phrase = re.sub(r"\'ll", " will", str(phrase))
    phrase = re.sub(r"\'t", " not", str(phrase))
    phrase = re.sub(r"\'ve", " have", str(phrase))
    phrase = re.sub(r"\'m", " am", str(phrase))
    return phrase

# Preprocess text
def preprocess(text_column):
    my_list = []
    # tqdm is for printing the status bar
    for sentance in tqdm(text_column.values):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e.lower() for e in sent.split())
        my_list.append(sent.lower().strip())
    
    return my_list