import os
from pathlib import Path
from src.embeddings import get_embeddings 

def ingest():
    """
    Scans the data/incidents directory, parses each file's metadata and content,
    generates a vector embedding, and returns a list of document dictionaries.
    """
    documents = []
    # Target the relative directory path safely using pathlib
    incidents_dir = Path("data/incidents")
    
    # Verify the directory actually exists before running
    if not incidents_dir.exists():
        print(f"Error: The directory {incidents_dir} does not exist.")
        return documents

    # Loop through every .txt file inside the directory
    for file_path in incidents_dir.glob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Initialize empty fields for parsing
            date = ""
            service = ""
            summary = ""
            
            # Parse line by line based on headers
            for line in lines:
                cleaned_line = line.strip()
                if cleaned_line.startswith("DATE:"):
                    date = cleaned_line.replace("DATE:", "").strip()
                elif cleaned_line.startswith("SERVICE:"):
                    service = cleaned_line.replace("SERVICE:", "").strip()
                elif cleaned_line.startswith("SUMMARY:"):
                    summary = cleaned_line.replace("SUMMARY:", "").strip()

            # Construct the comprehensive text layout that will be vectorized
            # Including metadata in the text block helps the model match context better
            text_to_embed = f"Date: {date} | Service: {service} | Summary: {summary}"
            
            # Generate the 384-dimensional vector embedding
            vector_embedding = get_embeddings([text_to_embed])[0]
            
            # Package everything into a structured dictionary object
            doc_object = {
                "file_name": file_path.name,
                "metadata": {
                    "date": date,
                    "service": service
                },
                "original_text": summary,
                "embedding": vector_embedding
            }
            
            documents.append(doc_object)
            
        except Exception as e:
            print(f"Failed to process file {file_path.name}: {e}")
            
    return documents
