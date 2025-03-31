import numpy as np
from transformers import pipeline
# Using Hugging Face GPT2 Model
# huggingface_model = pipeline("text-generation", model = "gpt2") -> if using hugging face
import openai 
import os

features = {
    "absolute": {
        "tempo": 120,
        "audioDuration": 150.0
    },
    "relative": {
        "rms": [0.21, 0.31, 0.28, 0.25, 0.30],
        "spectral_centroid": [1500, 1800, 1700, 1650, 1750],
        "zero_crossing_rate": [0.02, 0.015, 0.018, 0.016, 0.017],
        "concepts": ["calm", "energetic", "melancholic", "upbeat", "mysterious"]
    }
}

def generate_prompt(features):
    """
    We can generate a text prompt from the extracted audio features.
    :param features: Extracted feature dictionary from Tim's featureextraction.py
    :return: A prompt for the LLM
    """

    rms_values = np.array(features['relative']['rms'], dtype=np.float32)
    spectral_centroid_values = np.array(features['relative']['spectral_centroid'], dtype=np.float32)
    zero_crossing_rate_values = np.array(features['relative']['zero_crossing_rate'], dtype=np.float32)

    concepts_list = features['relative'].get('concepts', [])

    prompt = f"""
    The music has a tempo of {features['absolute']['tempo']} BPM and a duration of {features['absolute']['audioDuration']} seconds.
    It contains the following features:

    - RMS values (first 5): {rms_values[:5].tolist()}

    - Spectral Centroid (first 5): {spectral_centroid_values[:5].tolist()}

    - Zero Crossing Rate (first 5): {zero_crossing_rate_values[:5].tolist()}

    - Chord Concepts (first 5): {concepts_list[:5]}

    Describe a visual scene inspired by these musical features.
    """

    return prompt

def query(prompt):
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a creative AI that generates visual scene descriptions based on music features."},
                {"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return result.choices[0].message.content

prompt = generate_prompt(features)
print(prompt)

result = query(prompt)
print(result)

# Better huggingface model:
# huggingface_model = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
# def query2(prompt):
#   response = huggingface_model(prompt, max_new_tokens=150, num_return_sequences=1, truncation=True)
#   return response[0]["generated_text"]

