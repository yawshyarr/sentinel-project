import numpy as np
from src.embeddings import get_embeddings

def main ():
    # Example usage of the get_embeddings function
    sentences = [
        "Hello, world!", "This is a test sentence.",
        "Embeddings are useful for many NLP tasks.",
        "Let's see how well this model performs.",
        ]
    embeddings = get_embeddings(sentences)
    
    print("Generating vector embeddings...")
    
    # 2. Embed each sentence and look at the math layout
    for sentence in sentences:
        vector = get_embeddings([sentence])[0]
        
        # Print the text and its "shape" (length of the list of numbers)
        print(f"\nText: '{sentence}'")
        print(f"Vector Length (Dimensions): {len(vector)}")
        print(f"First 5 numbers of the vector: {vector[:5]}")

if __name__ == "__main__":
    main()