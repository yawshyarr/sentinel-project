from src.embeddings import get_embeddings
from src.similarity import cosine_similarity

def main():
    # Example usage
    texts = ["Hello world", "Hi there",
             "Goodbye world", "Farewell",
             "Hello everyone", "Hi all",
             "Goodbye everyone", "Farewell all"]
    
    print("1. Generating embeddings...")
    vectors = [get_embeddings([s])[0] for s in texts]
    
    print("\n2. Computing Semantic Similarity Matrix...")
    print("-" * 60)
    
    # Outer loop selects the first sentence
    for i in range(len(texts)):
        # Inner loop selects the second sentence to compare against
        for j in range(len(texts)):
            # Skip comparing a sentence to itself to keep the logs clean
            if i >= j:
                continue
                
            score = cosine_similarity(vectors[i], vectors[j])
            
            print(f"Sentence A: '{texts[i]}'")
            print(f"Sentence B: '{texts[j]}'")
            print(f"Similarity Score: {score:.4f}")
            print("-" * 60)

if __name__ == "__main__":
    main()