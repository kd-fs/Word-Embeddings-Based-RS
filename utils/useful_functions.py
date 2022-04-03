""" 
@author: Darlin KUAJO
Ce module contient les fonctionnalités utiles au developpement de Word2Vec-Based Recommender System"""

import os
import nltk
import string
import numpy as np
from tqdm.notebook import tqdm

FR_STOPWORD = nltk.corpus.stopwords.words('french')
PATH_DATA = ".\\dataset.zip"
PATH_METADATA = ".\\dataset\\metadata.json"

def remove_punctuation(texte):
    """ Cette fonction permet de supprimer les caractères de ponctuation contenus dans un texte
    
    texte : str
    
    Valeur de retour : str
    """
    
    result = "".join( [ch for ch in texte if ch not in string.punctuation])
    
    return result

def tokenizer(texte):
    """ Cette fonction permet de tokéniser un texte 
    
    texte : str
    
    Valeur de retour : list(str)
    """
    
    words = texte.split()
    
    return words

def remove_stopwords(tokens_list):
    """ Cette fonction permet de supprimer les mots vides de la langue anglaise contenus dans une liste de tokens 
    
    tokens_list : list(str)
     
     Valeur de retour : list(str)
    """
    
    result = [word for word in tokens_list if word not in FR_STOPWORD]
    
    return result

def clear_description(description):
    """Cette fonction permet de prétraiter la description textuelle d'un produit
    
    Les opérations de prétraitement sont:
    - Conversion de la description en minuscule
    - Suppression des caractères de ponctuation
    - Tokénisation
    - Suppression des mots vides
    
    description : str
    
    Valeur de retour : list(str)
    """
    
    description = description.lower()
    description = remove_punctuation(description)
    description = tokenizer(description)
    description = remove_stopwords(description)
    
    return description