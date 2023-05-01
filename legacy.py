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