import tensorflow_hub as hub

# Import the Universal Sentence Encoder's TF Hub module
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)

# Function to run the model and return embeddings
def embed(input):
    return model(input)