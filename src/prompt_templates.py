def get_grounded_prompt(context_blocks: list, user_query: str) -> str:
    """
    Constructs a strict grounding prompt. Forces the model to use provided 
    source files and explicitly cite the filename, or refuse if data is missing.
    """
    # Format the retrieved documents cleanly for the AI's context window
    formatted_context = ""
    for doc in context_blocks:
        formatted_context += f"--- START FILE: {doc['file_name']} ---\n"
        formatted_context += f"DATE: {doc['metadata']['date']}\n"
        formatted_context += f"SERVICE: {doc['metadata']['service']}\n"
        formatted_context += f"SUMMARY: {doc['original_text']}\n"
        formatted_context += f"--- END FILE ---\n\n"

    # Strict professional formatting rules inspired by enterprise prompt engineering patterns
    prompt = (
        "You are an expert site reliability engineering incident analyst. Your objective is to answer the user's question "
        "using ONLY the structured file data wrapped in 'START FILE' and 'END FILE' markers below.\n\n"
        f"Available Factual Data:\n{formatted_context}"
        "STRICT OPERATIONAL RULES:\n"
        "1. If the provided data contains the answer, explain the resolution clearly.\n"
        "2. You MUST explicitly state the filename(s) you used to find the answer (e.g., '[Source: incident_01.txt]').\n"
        "3. If the available data does not contain the answer, or if the context block is empty, you must respond with "
        "exactly this phrase: 'I do not have access to operational logs regarding that event.' Do not attempt to guess or use outside knowledge.\n\n"
        f"User Question: {user_query}\n\n"
        "Grounded Response:"
    )
    return prompt
