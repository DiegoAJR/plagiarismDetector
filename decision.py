from preprocess import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import seaborn as sns
from preprocess import preprocessing
from preparation import preparation


def print_confussion_matrix(y_test, y_pred):
    sns.heatmap((confusion_matrix(y_test,y_pred)), annot=True, fmt="d",cmap="crest")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()


def build_word_frequency_histogram(preprocessed_str):
    wordfreq = {}
    for word in preprocessed_str.split():
        if word not in wordfreq:
            wordfreq[word] = 0
        wordfreq[word] += 1
    return wordfreq


def decision():
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
    
    
    print("len processed suspicious: ", len(processed_original_texts))
    # Comparing suspicious text with original texts
    for processed_suspicious_text in processed_suspicious_texts:
        print("Suspicious text: ", suspicious_text)
        plagiarized_check = False
        for i, processed_original_text in enumerate(processed_original_texts):
            unigram_result, trigram_result = preparation(processed_suspicious_text, processed_original_text)

            # If similarity is greater than 0.5 in unigrams and more , then plagiarism is detected
            if unigram_result > 0.15 and trigram_result > 0.05:
                print("Plagiarism detected in file: ", original_texts[i])
                print("⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️")
                print("\n")
                
                if dictPlag.get(suspicious_text) is None:
                    dictPlag[suspicious_text] = [[original_texts[i],unigram_result,trigram_result]]
                else:
                    dictPlag[suspicious_text].append([original_texts[i], unigram_result,trigram_result])
                    
                
                if not plagiarized_check:
                    system_results.append(1)
                    plagiarized_check = True
            
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
            # When there is no plagiarism, append 0 to system results
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
    print_confussion_matrix(actual_results, system_results)

    # MATCH: %match, indexes, textmatch, OGFileName


if __name__ == "__main__":
    decision()