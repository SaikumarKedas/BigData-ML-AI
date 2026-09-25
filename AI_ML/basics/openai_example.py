import os
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env file
load_dotenv()

# Create an OpenAI client instance
client = OpenAI()


# Ask OpenAI a question and return the answer
def ask_openai(question):
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input= question
    )
    return response.output_text

question = input("Ask OpenAI: ")
answer = ask_openai(question)

print("\nOpenAI: {}".format(answer))