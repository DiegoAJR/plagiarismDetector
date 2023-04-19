from preprocess import preprocessing
from nltk.util import ngrams

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

def preparation():
    text1 = "legit.txt"
    text2 = "plagiarized.txt"
    legitpreprocessed, plagiarizedpreprocessed = preprocessing(text1, text2)
    paragraphs = [legitpreprocessed, plagiarizedpreprocessed]

    # N-grams
    unigrams = create_n_grams(legitpreprocessed.split(" "), plagiarizedpreprocessed.split(" "), 1)
    trigrams = create_n_grams(legitpreprocessed.split(" "), plagiarizedpreprocessed.split(" "), 3)
    

    # Embeddings
    embeddings_uni = create_embeddings(unigrams)
    embeddings_tri = create_embeddings(trigrams)

    build_embeddings(embeddings_uni, unigrams, paragraphs)
    build_embeddings(embeddings_tri, trigrams, paragraphs)

    print(unigrams)
    # print(trigrams)
    print(embeddings_uni)
    # print(embeddings_tri)
    

if __name__ == "__main__":
    preparation()