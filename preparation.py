from preprocess import preprocessing
from nltk.util import ngrams
from sklearn import metrics
from sklearn.metrics import pairwise, confusion_matrix
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import seaborn as sns


def printConfussionMatrix(y_test, y_pred):
    sns.heatmap((confusion_matrix(y_test,y_pred)), annot=True, fmt="d",cmap="crest")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()



def create_n_grams(p1,p2,n):
    ngrams1 = ngrams(p1, n)
    ngrams2 = ngrams(p2, n)
    nset1 = set(ngrams1)
    nset2 = set(ngrams2)
    nsets = nset1.union(nset2)
    
    return nsets

def create_embeddings(uniques):
    return [[0 for _ in range(len(uniques))] for _ in range(2)]

def build_embeddings(embeddings,uniques,paragraphs):
    for p in range(2):
        for i,unique in enumerate(uniques):
            s = " ".join(list(unique))
            if s in paragraphs[p]:
                embeddings[p][i] = 1

def find_similarities(text1, text2):
    # Reading files
    with open(text1, "r") as legit:
        legit = legit.read()
    with open(text2, "r") as plagiarized:
        plagiarized = plagiarized.read()
    
    matcher = SequenceMatcher(None, legit, plagiarized)
    matching_blocks = matcher.get_matching_blocks()
    similarities = []
    for block in matching_blocks:
        start = block.a
        end = block.a + block.size
        similarity = legit[start:end]
        position = block.b
        similarities.append((similarity, position))

    return [similarity for similarity in similarities if len(similarity[0]) > 10]

def build_word_frequency_histogram(preprocessed_str):

    wordfreq = {}
    for word in preprocessed_str.split():
        if word not in wordfreq:
            wordfreq[word] = 0
        wordfreq[word] += 1
    return wordfreq

def preparation(processed_suspicious_text, processed_original_text):
    
    paragraphs = [processed_suspicious_text, processed_original_text]

    # N-grams
    unigrams = create_n_grams(processed_suspicious_text.split(" "), processed_original_text.split(" "), 1)
    trigrams = create_n_grams(processed_suspicious_text.split(" "), processed_original_text.split(" "), 3)
    

    # Embeddings
    embeddings_uni = create_embeddings(unigrams)
    embeddings_tri = create_embeddings(trigrams)

    build_embeddings(embeddings_uni, unigrams, paragraphs)
    build_embeddings(embeddings_tri, trigrams, paragraphs)

    
    # Cosine similarity para unigramas
    unigram_result = pairwise.cosine_similarity(embeddings_uni)[0,1]
    print("Result Unigram: " , unigram_result)
    trigram_result = pairwise.cosine_similarity(embeddings_tri)[0,1]
    print("Result Trigram: " , trigram_result)
    return unigram_result, trigram_result


if __name__ == "__main__":
    preparation()