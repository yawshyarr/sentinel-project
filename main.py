from src.llm import generate

def main():
    # Define a test prompt
    prompt = "Explain why the sky is blue in exactly two sentences."
    
    # Specify the exact model you downloaded in Day 1 or Day 2
    # Examples: 'llama3.2:3b', 'llama3.2:1b', 'qwen2.5:7b'
    chosen_model = "qwen2.5:1.5b" 
    
    print(f"Sending prompt to {chosen_model}...")
    
    # Call your reusable function
    result = generate(prompt, model=chosen_model)
    
    # Print the output
    print("\n--- Model Response ---")
    print(result)

if __name__ == "__main__":
    main()
