"""
This file contains all code related to loading LLMs and prompts!
Using Gemini: https://ai.google.dev/gemini-api/docs/quickstart?lang=python
"""

from google import genai
from util import get_api_key


######################################## GEMINI ########################################
def get_gemini_client(api_key):
    return genai.Client(api_key=f"{api_key}")

def gemini(prompt, client=None):
    if client is None:
        default_api_key = get_api_key('gemini')
        client = get_gemini_client(default_api_key)
    response = client.models.generate_content(
      model="gemini-2.0-flash", 
      contents=[prompt],
      config={
        "max_output_tokens": 77,  # Set the maximum number of output tokens (77 is max for AnimateDiff tokenizer)
        "temperature": 0.7,       # Optional: Controls randomness
        "top_p": 0.95             # Optional: Nucleus sampling
      }
    )
    return response.text






