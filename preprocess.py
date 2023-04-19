# Used Packages
import nltk
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
import re

def preprocessing(file):
    
    # Reading files
    with open(file, "r") as document:
        document = document.read()

    # Remove prepositions, articles, etc
    from gensim.parsing.preprocessing import remove_stopwords

    filtered_doc = remove_stopwords(document)

    # Turn to lowercase
    filtered_doc = filtered_doc.lower()

    # Define the pattern to match
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
    
    return doc_par