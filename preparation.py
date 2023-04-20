from preprocess import preprocessing
from nltk.util import ngrams
from sklearn import metrics
from sklearn.metrics import pairwise, confusion_matrix
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
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
    #with open(file, "r") as file_str:
    #    file_str = file_str.read()
    # tokens = nltk.word_tokenize(preprocessed_str.lower()) # Convert to lowercase and tokenize
    # freq_dist = FreqDist(tokens)
    # return freq_dist
    # build dictionary of words and their frequency
    wordfreq = {}
    for word in preprocessed_str.split():
        if word not in wordfreq:
            wordfreq[word] = 0
        wordfreq[word] += 1
    return wordfreq

def preparation():
    dictPlag = {}
    
    # Preprocessing original texts
    original_texts = [file for file in os.listdir("original_files") if os.path.isfile(os.path.join("original_files", file))]
    processed_original_texts = []
    for original_text in original_texts:
        processed_original_texts.append(preprocessing("original_files/" + original_text))


    print("Starting plagiarism detection...")
    print("\n")
    # Preprocessing suspicious texts
    suspicious_texts = [file for file in os.listdir("suspicious_files") if os.path.isfile(os.path.join("suspicious_files", file))]
    processed_suspicious_texts = []
    print(suspicious_texts)
    for suspicious_text in suspicious_texts:
        processed_suspicious_texts.append(preprocessing("suspicious_files/" + suspicious_text))

    actual_results = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    system_results = []
    
    #all_similarities = []

    print("len processed suspicious: ", len(processed_original_texts))
    # Comparing suspicious text with original texts
    for processed_suspicious_text in processed_suspicious_texts:
        print("Suspicious text: ", suspicious_text)
        plagiarized_check = False
        for i, processed_original_text in enumerate(processed_original_texts):
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

            pointers = False
            # If similarity is greater than 0.5 in unigrams and more , then plagiarism is detected
            if unigram_result > 0.15 and trigram_result > 0.05:
                print("Plagiarism detected in file: ", original_texts[i])
                pointers = True
                print("⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️")
                print("\n")
                
                if dictPlag.get(suspicious_text) is None:
                    dictPlag[suspicious_text] = [[original_texts[i],unigram_result,trigram_result]]
                else:
                    dictPlag[suspicious_text].append([original_texts[i], unigram_result,trigram_result])
                    
                
                if not plagiarized_check:
                    system_results.append(1)
                    plagiarized_check = True
            
            if pointers:
                # similarities = find_similarities(suspicious_text, original_texts[i])
                # all_similarities.append([original_texts[i], similarities])
                # for similarity, position in similarities:
                #     print(f"Similarity: '{similarity}', Position: {position}")\


                histogram = build_word_frequency_histogram(processed_suspicious_text)
                categories = list(histogram.keys())[:20]
                frequencies = list(histogram.values())[:20]

                plt.bar(categories, frequencies)            
                plt.xlabel('Categories')
                plt.ylabel('Frequency')
                plt.title('Category Frequency Histogram')
                plt.xticks(rotation=90)
                #plt.show()

        if not plagiarized_check:
            system_results.append(0) 
    
    print(dictPlag)  
    print(f'SYS {system_results}')
    print(f'Actual {actual_results}')
    fpr, tpr, thresholds = metrics.roc_curve(actual_results, system_results, pos_label=1)
    
   
    # RECORDAR ARREGLAR LO DEL ROOOOC
    
    print("False Positive Rate: ", fpr)
    print("True Positive Rate: ", tpr)
    print("AUC:", metrics.auc(fpr, tpr))
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.title('ROC curve')
    plt.xlabel("Error")
    plt.ylabel("True Positive Rate")
    plt.show()

    

    
    
    
    #print("Similarities: " , all_similarities)
    # MATCH: %match, indexes, textmatch, OGFileName


if __name__ == "__main__":
    preparation()