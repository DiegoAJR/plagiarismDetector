# Used Packages
from sklearn.metrics import pairwise
from model import embed

'''
Main preparation function that calculates the cosine similarity of two pre-processed texts
by creating unigrams and trigrams.

Receives two pre-processed texts
Retruns the result of applying the cosine difference to both sets of unigrams and trigrams
'''
def preparation(suspicious_embedding, original_embeddings, suspicious_text):
    
    suspicious_word_count_plagiarims = 0

    # Cosine similarity calculation sentence by sentence
    for i in range(len(original_embeddings)):
        for j in range(len(suspicious_embedding)):
            cosine_result = pairwise.cosine_similarity(original_embeddings[i],suspicious_embedding[j])
            if cosine_result[0][0] > 0.75:
                suspicious_word_count_plagiarims += len(suspicious_text[j].split(" "))* cosine_result[0][0]


    return suspicious_word_count_plagiarims