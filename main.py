from src.ingestion import ingest
from src.retrieve import retrieve
from src.prompt_templates import get_grounded_prompt
from src.llm import generate

def run_rag_pipeline(query: str, database: list, model_name: str, threshold: float = 0.40):
    print(f"\n🔎 USER QUERY: '{query}'")
    
    # 1. Retrieve the top 3 best matching files
    top_docs = retrieve(query, database, top_k=3)
    
    # Check if we have any results at all, and inspect the absolute top score
    if not top_docs or top_docs[0]["similarity_score"] < threshold:
        print(f"Search confidence below threshold ({top_docs[0]['similarity_score']:.4f} < {threshold}). Bypassing LLM.")
        print("AI RESPONSE: I do not have access to operational logs regarding that event.")
        print("=" * 70)
        return

    # 2. If it passes the threshold, look up file references and build prompt
    print(f"Context verified (Top Match Score: {top_docs[0]['similarity_score']:.4f})")
    grounded_prompt = get_grounded_prompt(top_docs, query)
    
    # 3. Query your local offline LLM
    print(f"🧠 Prompting local model ({model_name})...")
    response = generate(grounded_prompt, model=model_name)
    
    print("\nAI RESPONSE:")
    print(response)
    print("=" * 70)

def main():
    # Set model based on your system RAM architecture (e.g., 'llama3.2:1b' or 'llama3.2:3b')
    chosen_model = "llama3.2:3b" 
    
    print("1. Ingesting and embedding local incident directory...")
    database = ingest()
    print(f"   Successfully indexed {len(database)} documents.\n")
    
    print("2. Launching Grounded Pipeline Evaluation...")
    print("=" * 70)
    
    # Test 1: Legitimate query (Should generate response + explicit source citation)
    run_rag_pipeline("Tell me about the stripe webhook payment issue", database, chosen_model)
    
    # Test 2: Unrelated query (Should be stopped instantly by threshold or prompt rule)
    run_rag_pipeline("What is the current outdoor weather temperature in London?", database, chosen_model)

if __name__ == "__main__":
    main()
