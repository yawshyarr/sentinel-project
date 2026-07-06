import numpy as np
from src.embeddings import get_embedding    
from src.similarity import cosine_similarity
from src.llm import generate

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast, dark-colored fox leaps over a sleepy canine.",  
    "An unrelated sentence about a different topic.",
]

#embeddings 
print("Generating embeddings...")
embeddings = [get_embedding(text) for text in texts]

#question from user 
print("\nUser Question: ")
user_query = input("Please enter your question: ")

#embedding for user query
user_embedding = get_embedding(user_query)

#similarity scores
print("\nCalculating similarity scores...")
best_score = -1
best_factor = None

for i , j in zip(range(len(texts)), embeddings):
    score = cosine_similarity(user_embedding, j)
    print(f"Similarity score with '{texts[i]}': {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_factor = texts[i]

#rag prompt
print("\nGenerating response using RAG...")
rag_prompt = f"Based on the most similar text: '{best_factor}', answer the question: {user_query}"

#model 
chosen_model = "qwen2.5:1.5b"  # Change this to your preferred model if needed
response = generate(rag_prompt, model=chosen_model)
print("\nResponse from the model:")
print(response)

if __name__ == "__main__":
    pass  # The main execution is handled above, no additional code needed here.