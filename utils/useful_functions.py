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

def imUrl_to_image_html_width100(imPath):
    """Cette fonction permet de convertir l'adressse de l'image d'un produit en
    balise HTML nécessaire pour afficher l'image avec une largeur de 100px
    
    imUrl : str
    
    Valeur de retour : str
    """
    
    return '<img src="'+ os.path.join("dataset", imPath) + '" width="100px" >'

def sim_cosine(item_embedding1, item_embedding2):
    """Cette fonction calcul la similarité de cosinus entre les deux items embeddings item_embedding1 et  item_embedding2
    
    item_embedding1 : numpy.ndarray
    item_embedding2 : numpy.ndarray
    
    Valeur de retour : float
    """
    
    cosine = np.dot(item_embedding1, item_embedding2) / (np.linalg.norm(item_embedding1) * np.linalg.norm(item_embedding2))
    
    return cosine

def mean_pooling(matrix):
    """ Cette fonction calcul la moyenne colonne à colonne de la matrice matrix
    
    matrix : numpy.ndarray
    
    Valeur de retour : numpy.ndarray
    """
    
    return np.mean(matrix, axis=0)

def itemLT(asin, model, data):
    """ Cette fonction permet de renvoyer le item embedding de l'item asin
    
    asin : str
    model : gensim.models.word2vec.Word2Vec
    data : pd.DataFrame
    
    Valeur de retour : numpy.ndarray
    """
    
    description = data[data['ID'] == asin]
    description.reset_index(inplace=True, drop=True)
    description = description.iloc[0]['title']
    tokens_list = clear_description(description)
    item_embedding = []
    for i in range(len(tokens_list)):
        item_embedding.append(model.wv[tokens_list[i]])
    item_embedding = np.array(item_embedding)
    item_embedding = mean_pooling(item_embedding)
    
    return item_embedding

def top_n_recommendation(asin, n, model, data, similarity_mesure):
    """Cette fonction renvoie la liste des n premiers produits les plus similaires au produit asin en guise de liste de recommandation top-n,
    selon l'ordre décroissant des scores de similarité de ces produits par rapport au produit asin
    
    asin : str
    n : int
    model : gensim.models.word2vec.Word2Vec
    data : pd.DataFrame
    similarity_mesure : function
    
    n > 0
        
    Valeur de retour : list((float, str))
    """
    
    i=0
    recommendations_list = []
    item_embedding1 =  itemLT(asin, model, data)
    
    
    # Calcul des scores de similarité entre le produit asin et tous les autres produits
    for i in tqdm(data.index):
        if data.iloc[i]['ID'] != asin:
            item_embedding2 = itemLT(data.iloc[i]['ID'], model, data)
            recommendations_list.append((similarity_mesure(item_embedding1, item_embedding2), data.iloc[i]['ID']))
    
    # Tri des produits par ordre décroissant des scores de similarité 
    recommendations_list.sort(reverse=True)
    # Extraction des n premiers produits
    recommendations_list = recommendations_list[0:n]
    
    return recommendations_list