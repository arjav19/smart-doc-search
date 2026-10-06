"""Day 0 smoke test: can we talk to Gemini at all?"""
import os
from dotenv import load_dotenv
from google import genai


load_dotenv()                                   # reads GEMINI_API_KEY from .env
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
chat = client.chats.create(model="gemini-3.8-flash")
response = chat.send_message("Say hello in one short sentence.")

print("\n--- Response ---")
print(response.text)
