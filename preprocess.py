# Used Packages
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download("wordnet")
from gensim.parsing.preprocessing import remove_stopwords

'''
Preprocessing of original text files. Preprocessing involves the removal of stopwords, 
turning the text to lower case, removal of special characters and lemmatization.

Receives the name of the text file to preprocess.
Returns the preprocessed text as a string.
'''
def preprocessing(file, test = False):
    
    if not test:
        # Reading files
        with open(file, "r", encoding="utf_8") as document:
            document = document.read()
    else:
        document = file

    # Remove prepositions, articles, etc
    filtered_doc = remove_stopwords(document)

    # Turn to lowercase
    filtered_doc = filtered_doc.lower()

    # Define the pattern to match everything that is a special character
    pattern = r"[^\w\s]"

    # Remove special characters
    doc_par = re.sub(pattern, "", filtered_doc)

    # Spliting per sentence
    doc_par = doc_par.split(" ")

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    for i in range(len(doc_par)):
        doc_par[i] = lemmatizer.lemmatize(doc_par[i])

    # Join the sentences
    doc_par = " ".join(doc_par)
    
    # Return preprocessed file as a string
    return doc_par


