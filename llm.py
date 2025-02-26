"""
This file contains all code related to loading LLMs and prompts!
Using Gemini: https://ai.google.dev/gemini-api/docs/quickstart?lang=python
"""

from google import genai

def get_gemini_client(api_key):
    return genai.Client(api_key=f"{api_key}")

def gemini(prompt, client):
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=[prompt]
    )
    return response.text






