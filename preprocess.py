# Used Packages
import nltk
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
import re

def preprocessing(file1, file2):
    
    # Reading files
    with open(file1, "r") as legit:
        legit = legit.read()
    with open(file2, "r") as plagiarized:
        plagiarized = plagiarized.read()

    # Remove prepositions, articles, etc
    from gensim.parsing.preprocessing import remove_stopwords

    filtered_legit = remove_stopwords(legit)
    filtered_plagiarized = remove_stopwords(plagiarized)

    # Turn to lowercase
    filtered_legit = filtered_legit.lower()
    filtered_plagiarized = filtered_plagiarized.lower()

    # Define the pattern to match
    pattern = r"[^\w\s]"


    # Remove special characters
    legit_par = re.sub(pattern, "", filtered_legit)
    plagiarized_par = re.sub(pattern, "", filtered_plagiarized)

    # Spliting per sentence
    legit_par = legit_par.split(" ")
    plagiarized_par = plagiarized_par.split(" ")

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    for i in range(len(legit_par)):
        legit_par[i] = lemmatizer.lemmatize(legit_par[i])

    for i in range(len(plagiarized_par)):
            plagiarized_par[i] = lemmatizer.lemmatize(plagiarized_par[i])

    # Join the sentences
    legit_par = " ".join(legit_par)
    plagiarized_par = " ".join(plagiarized_par)
    
    return legit_par, plagiarized_par