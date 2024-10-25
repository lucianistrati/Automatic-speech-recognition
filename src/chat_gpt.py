from dotenv import load_dotenv
import openai
import os

# Load environment variables from a .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPEN_AI_API_KEY")


def chat_gpt(prompt: str, model_name: str = 'gpt-3.5-turbo') -> str:
    """Generate a response from the OpenAI GPT model based on the provided prompt.

    Args:
        prompt (str): The input prompt for the model.
        model_name (str): The name of the model to use. Default is 'gpt-3.5-turbo'.

    Returns:
        str: The generated response from the model.

    Raises:
        ValueError: If the prompt is empty.
        Exception: If there is an issue with the OpenAI API call.
    """
    # Check if the prompt is not empty
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    try:
        # Send a call to the OpenAI API
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )

        # Extract and return the generated text
        generated_text = response['choices'][0]['message']['content']
        return generated_text

    except Exception as e:
        # Handle any exceptions that occur during the API call
        raise Exception(f"An error occurred while communicating with OpenAI API: {e}")


# Example usage
if __name__ == "__main__":
    user_prompt = "What are the benefits of using artificial intelligence?"
    try:
        response = chat_gpt(user_prompt)
        print("GPT-3.5 Response:", response)
    except Exception as e:
        print("Error:", e)
