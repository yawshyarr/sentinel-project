import numpy as np

def cosine_similarity(vec1:np.ndarray, vec2:np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.
    Cosine similarity is defined as the dot product of the vectors divided by the product of their magnitudes.
    """
    dot_product = np.dot(vec1, vec2)

    "magnitude of the vectors"
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # Avoid division by zero; return 0 similarity if either vector is zero
    
    return dot_product / (norm_vec1 * norm_vec2)