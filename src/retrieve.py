import numpy as np
from src.similarity import cosine_similarity
from src.embeddings import get_embeddings

def retrieve(query: str, documents: list, top_k: int = 5) -> list:

    scored_documents = []

    #generate embedding for the query
    query_embedding = get_embeddings(query)

    # Loop through each document to compute similarity
    for doc in documents:
        score = cosine_similarity(query_embedding, doc["embedding"])
        scored_documents.append({
            "file_name": doc["file_name"],
            "metadata": doc["metadata"],
            "original_text": doc["original_text"],
            "similarity_score": score
        })

    # Sort documents by similarity score in descending order
    scored_documents.sort(key = lambda x: x["similarity_score"], reverse=True)

    # Return the top k documents
    return scored_documents[:top_k]