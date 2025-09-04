def load_prompt(prompt_file):
    """Load a prompt template from a text file."""
    import os
    
    # Get the directory where this file is located
    current_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(current_dir, "prompts", prompt_file)
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()
