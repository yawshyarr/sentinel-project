from sentence_transformers import SentenceTransformer

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
    return model.encode(texts)