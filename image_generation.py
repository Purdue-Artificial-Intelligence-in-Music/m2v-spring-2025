"""
This file contains methods to generate images from features and save them to an output folder.
"""

from diffusers import StableDiffusionPipeline
from util import get_prompt
import os

NEGATIVE_PROMPT = "humans, shapes, letters, low quality, blurry, painting frame"

def load_default_pipe() -> StableDiffusionPipeline:
    """
    Loads a default image generation pipeline.
    @return: the default Stable Diffusion Pipeline with image checker disabled
    """
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
    pipe.to("cuda")
    return pipe

def features_to_image_prompts(features: dict) -> list[str]:
    """
    Generate prompts based on the input music features.
    @param features: a dictionary containing the extracted features
    @return: a list of the generated prompts
    """
    # Unpack music info
    tempo = features["absolute"]["tempo"]
    concepts = features["relative"]["concepts"]

    print(concepts)

    # Generate prompts
    prompts = []
    for _, concept in enumerate(concepts):
        prompt = get_prompt('image')
        prompt = prompt.format(concept=concept, tempo=tempo)
        prompts.append(prompt)

    return prompts

def generate_images(prompts: list[str], output_folder: str, pipe: StableDiffusionPipeline = load_default_pipe()):
    """
    Generate images based on the music features extracted from the audio file.
    @param prompts: a list of prompts to generate, one image per prompt
    @param output_folder: the folder where the generated images will be saved
    @param pipe: the pipeline to use for generating images
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = []
    for i, prompt in enumerate(prompts):
        image = pipe(prompt, negative_prompt=NEGATIVE_PROMPT).images[0]
        image_name = f"frame_{i:03d}.png"
        image_path = os.path.join(output_folder, image_name)
        image.save(image_path)