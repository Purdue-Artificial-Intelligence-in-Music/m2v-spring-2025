import google.generativeai as genai
import os

# Set your API key
with open('geminiKey.txt', 'r') as f:
    key = f.readlines()[0]
    print(key)
    os.environ["API_KEY"] = key

class GeminiWrapper:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')

    def generate_content(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text

