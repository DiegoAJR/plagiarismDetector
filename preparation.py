# Used Packages
from nltk.util import ngrams
from sklearn.metrics import pairwise
import numpy as np
from model import embed


'''
Divide the input paragraphs into unique n-grams as specified by the user parameter "n".

Receives two input paragraphs and an integer specifying the type of n-gram to be processed.
Returns a set with both texts spliced up into unique n-grams.
'''
def create_n_grams(p1,p2,n):

    # Create n-grams
    ngrams1 = ngrams(p1, n)
    ngrams2 = ngrams(p2, n)

    # Filter down to unique n-grams
    nset1 = set(ngrams1)
    nset2 = set(ngrams2)

    # Join both sets
    nsets = nset1.union(nset2)
    
    return nsets

'''
Creates a matrix full of 0-vectors of the length of the uniques set.

Receives a set of unique n-grams.
Returns a matrix full of 0-vectors.
'''
def create_embeddings(uniques):

    return [[0 for _ in range(len(uniques))] for _ in range(2)]

'''
Modifies the previously built matrixes from 0s to 1s for all the unique words in the texts.

Receives the embeddings matrix build on the "create_embeddings" function.
Does not return anything but it modifies the original matrix.
'''
def build_embeddings(embeddings,uniques,paragraphs):

    for p in range(2):
        for i,unique in enumerate(uniques):
            s = " ".join(list(unique))
            if s in paragraphs[p]:
                embeddings[p][i] = 1

'''
Main preparation function that calculates the cosine similarity of two pre-processed texts
by creating unigrams and trigrams.

Receives two pre-processed texts
Retruns the result of applying the cosine difference to both sets of unigrams and trigrams
'''


def preparation(processed_suspicious_text, processed_original_text):
    
    # paragraph_original = [processed_original_text]
    # paragraph_sus = [processed_suspicious_text]
    # Create unigrams and trigrams
    # unigrams = create_n_grams(processed_suspicious_text.split(" "), processed_original_text.split(" "), 1)
    # trigrams = create_n_grams(processed_suspicious_text.split(" "), processed_original_text.split(" "), 3)

    # paragraphs = processed_suspicious_text + processed_original_text
    results = []
    embeddings_original = []
    for i in range(len(processed_original_text)):
        embeddings_original.append(embed([processed_original_text[i]]))
    embeddings_sus = []
    for i in range(len(processed_suspicious_text)):
        embeddings_sus.append(embed([processed_suspicious_text[i]]))

    # print("Texts: ", processed_original_text,"--other--" ,processed_suspicious_text)

    for i in range(len(processed_original_text)):
        for j in range(len(processed_suspicious_text)):
            results.append(pairwise.cosine_similarity(embeddings_original[i],embeddings_sus[j]))
            # print(pairwise.cosine_similarity(embeddings_original[i],embeddings_sus[j]),":", processed_original_text[i],"|",processed_suspicious_text[j])  

    return np.mean(results), max(results)

    # # Create embeddings matrixes
    # embeddings_uni = create_embeddings(unigrams)
    # embeddings_tri = create_embeddings(trigrams)

    # # Fill out the embeddings matrix with the unique n-grams
    # build_embeddings(embeddings_uni, unigrams, paragraphs)
    # build_embeddings(embeddings_tri, trigrams, paragraphs)
    
    # Embeddings
    # embeddings_original = model.encode(processed_original_text, convert_to_tensor=True)
    # embeddings_sus = model.encode(processed_suspicious_text, convert_to_tensor=True)

    # print(type(embeddings_original), embeddings_original.shape)

