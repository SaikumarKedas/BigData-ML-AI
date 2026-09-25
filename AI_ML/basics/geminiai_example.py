import os
import time
from dotenv import load_dotenv
from google import genai


# Load environment variables from .env file
load_dotenv()

# Create an OpenAI client instance
client = genai.Client()


chat = client.chats.create(model = "gemini-3.6-flash")

# Ask Gemini a question and return the answer
def ask_gemini(question):
    for attempt in range(5):
        try:
            response = chat.send_message(question)
            return response.text
        except Exception as e:
            if "503" in str(e):
                print(f"Gemini is busy/high demand. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
            else:
                raise
    return "Gemini is currently unavailable."


question = input("\n Ask Gemini: ")
answer = ask_gemini(question)

print("\n Gemini: {}\n".format(answer))