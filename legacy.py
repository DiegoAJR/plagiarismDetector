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

