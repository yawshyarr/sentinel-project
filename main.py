from src.ingestion import ingest
from src.retrieve import retrieve   

def main():
    #ingest documents from the data/incidents directory
    documents = ingest()
    print(f"Ingested {len(documents)} documents.")
    if not documents:
        print("No ingested documents found. Exiting.")

    #query 
    query = ["Service outage affecting multiple users",
             "System downtime due to maintenance",
             "Unexpected service disruption",
             "Scheduled maintenance notification",
             "Service interruption impacting users"]

    for q in query:
        print(f"\nQuery: {q}")
        top_k_results = retrieve(q, documents, top_k=3)

        #looping through the top k results and printing them
        for idx, result in enumerate(top_k_results):
            print(f"Rank {idx + 1}:")
            print(f"File Name: {result['file_name']}")
            print(f"Metadata: {result['metadata']}")
            print(f"Original Text: {result['original_text']}")
            print(f"Similarity Score: {result['similarity_score']:.4f}\n")

if __name__ == "__main__":
    main()