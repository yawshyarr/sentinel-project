import ollama

def generate(prompt: str, model: str = "qwen2.5:1.5b") -> str:
    """
    Sends a prompt to the local Ollama server and returns the text response.
    Change the default model name if you are using a different one (like llama3.2:1b).
    """
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response']
    except Exception as e:
        return f"Error communicating with local Ollama server: {e}"
