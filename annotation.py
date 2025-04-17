import torch
from sympy.physics.units import temperature
# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from tqdm import tqdm
import json

def retrieve_user_preferences(input_data):
    """
    Extract user preferences and unpreferences from the input data.
    """
    preferences = re.search(r'User Preference:\s*((?:"[^"]*"(?:,\s*)?)+)', input_data)
    preferences = preferences.group(1) if preferences else None
    unpreferences = re.search(r'User Unpreference:\s*((?:"[^"]*"(?:,\s*)?)+)', input_data)
    unpreferences = unpreferences.group(1) if unpreferences else None
    # print(f"User Preferences: {preferences}")
    # print(f"User Unpreferences: {unpreferences}")
    return preferences, unpreferences

def extract_target_movie(input_data):
    """
    Extract the target movie from the input data.
    """
    # Use regex to find the target movie
    match = re.search(r'Whether the user will like the target movie "(.*?)"\?', input_data)
    if match:
        return match.group(1)
    else:
        return None

def extract_explanation(output_data):
    """
    Extract the explanation from the output data.
    """
    # Use regex to find the explanation
    match = re.search(r'Explanation:\s*(.*)', output_data, flags=re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None


def main():
    # Load the model and tokenizer to the GPU
    gpu_properties = torch.cuda.get_device_properties(0)
    total_memory_gb = gpu_properties.total_memory / (1024 ** 3)
    print(f"Total GPU Memory: {total_memory_gb:.2f} GB")
    if torch.cuda.is_available():
        print("Using GPU")

    # model_id = "Qwen/Qwen2.5-7B-Instruct"
    model_id = "/home/ubuntu/ANLP-FinalProject/pretrained_model/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, truncation=True, max_length=1024)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    # Define the pipeline
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, torch_dtype=torch.float16, return_full_text=False)

    INSTRUCTIONS = """You are a smart annotation assistant for a movie recommendation system.
    Data fields:
    - User Preferences: movies the user likes.
    - User Unpreferences: movies the user dislikes.
    - Recommend: target movie.
    - Output: exactly one value, either "Yes." (user will adopt) or "No." (user will reject).
    Task:  Using the provided Output value, generate exactly **one** concise sentence (less then 50 words) that explains why the user would or would not adopt the recommendation. Cite the key preference(s) or unpreference(s) that drove the decision.   
    Formatting rule (no exceptions):  
    - Do **not** repeat or echo the Output field.
    - Do **not** produce any additional text.
    - Output exactly:
        Explanation: "$YOUR_ANSWER_HERE".
"""

    # Load the annoation data
    with open('/home/ubuntu/ANLP-FinalProject/data/movie/sample_train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)


    with open("/home/ubuntu/ANLP-FinalProject/data/movie/sample_train_annotated.json", "w", encoding="utf-8") as f:
        for obj in tqdm(data):
        # Process each JSON object
            input_data = obj['input']
            output_data = obj['output']

            # Extract user preferences and unpreferences
            user_preferences, user_unpreferences = retrieve_user_preferences(input_data)
            if user_preferences is None:
                user_preferences = "user has no specific preferences"    
            if user_unpreferences is None:
                user_unpreferences = "user has no specific unpreferences"

            # Extract target movie
            target_movie = extract_target_movie(input_data)

            # Prepare the prompt
            prompt = (
                INSTRUCTIONS.strip()
                + "\n\n"
                f"User Preferences: {user_preferences}\n"
                f"User Unpreferences: {user_unpreferences}\n"
                f"Recommend: {target_movie}\n"
                f"Output: {output_data}\n\n"
                "Now produce the Explanation:"
                )
            # prompt = f"{INSTRUCTIONS}. For this user, User Preference: {user_preferences}, User Unpreference: {user_unpreferences}. Recommend:{target_movie}. Output: {output_data}."

            # Generate the explanation
            result = pipe(prompt, max_length=512, do_sample=False, temperature=1)
            # Print the generated explanation
            # print(result[0]['generated_text'])
            explanation = extract_explanation(result[0]['generated_text'])
            # Append the explanation to the original object
            obj['explanation'] = explanation
            # Write the updated object to the new JSON file
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")


    


if __name__ == "__main__":
    main()
