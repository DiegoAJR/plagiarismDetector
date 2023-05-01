# Used Packages
from sklearn.metrics import pairwise
import numpy as np
from model import embed

'''
Main preparation function that calculates the cosine similarity of two pre-processed texts
by creating unigrams and trigrams.

Receives two pre-processed texts
Retruns the result of applying the cosine difference to both sets of unigrams and trigrams
'''
def preparation(processed_suspicious_text, processed_original_text):
    
    results = []
    embeddings_original = []

    # Creation of embeddings using Universal Sentence Encoder
    for i in range(len(processed_original_text)):
        embeddings_original.append(embed([processed_original_text[i]]))
    embeddings_sus = []
    for i in range(len(processed_suspicious_text)):
        embeddings_sus.append(embed([processed_suspicious_text[i]]))

    # Cosine similarity calculation for each document
    for i in range(len(processed_original_text)):
        for j in range(len(processed_suspicious_text)):
            results.append(pairwise.cosine_similarity(embeddings_original[i],embeddings_sus[j]))

    return np.mean(results), max(results)


