from dotenv import load_dotenv
import openai
import os
load_dotenv()


# Set your OpenAI API key
openai.api_key = os.getenv("OPEN_AI_API_KEY")


def chat_gpt(prompt: str):
    model_name = 'gpt-3.5-turbo-1106'

    # Send a call to the OpenAI API
    response = openai.Completion.create(
        engine=model_name,
        prompt=prompt,
        max_tokens=150
    )

    # Extract and return the generated text
    generated_text = response['choices'][0]['text']
    return generated_text
