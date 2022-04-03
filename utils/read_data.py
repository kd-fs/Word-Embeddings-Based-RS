"""
@author: Darlin KUAJO
Ce module contient les fonctions nécessaires à la lecture des données.
"""

import pandas as pd

def getDF(path):
    """ Cette fonction permet la lecture des données depuis le fichier .json de chemin d'accès path
    
    path : str
    
    Valeur de retour : pd.DataFrame
    """
    
    iden = []
    title = []
    slug = []
    category =[]
    imPath = []
    for line in open(path, 'r', encoding='UTF-8'):
        item = eval(line.strip())
        iden.append(item['ID'])
        title.append(item['title'])
        slug.append(item['slug'])
        category.append(item['category'])
        imPath.append(item['imPath'])
    df = pd.DataFrame({'ID':iden, 'title':title, 'slug':slug, 'category':category, 'imPath':imPath})
    return df
    