"""
This script includes code to generate videos from features.

Todos:
TODO 0: Add imports for the existing code
TODO 1: Implement segmentation, adapt code to use it (i.e., create video prompt per music segment)
"""

import torch
import random
import time
import os
from PIL import Image
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video
import cv2
from vsrife import rife
import numpy as np
import vapoursynth as vs
from util import get_prompt

SCHEDULER_PARAMS = {
    "clip_sample": False, # Having this as True can make video become very fuzzy
    "timestep_spacing": "linspace", # "linspace", "log?"
    "beta_schedule": "linear",
    "steps_offset": 5
}

MODEL_PARAMS = {
    "negative_prompt": "low contrast, fuzzy, blinking, bad quality, low detail, pixelated, low resolution, fast, \
        blurry, distorted, unnatural, unrealistic, overexposed, underexposed, washed out colors. overly saturated, \
        jerky motion, poor lighting, poorly composed, artifacts, noise, oversharpened, \
        dull, flat lighting, cartoonish, ugly, text, watermark,",
    "guidance_scale": 6,
    "num_inference_steps": 20,
    "fps": 8# I don't think we can change this. 8 is the default for AnimateDiff
}

def load_default_pipe():
    # Load the motion adapter
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    # load SD 1.5 based finetuned model
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=SCHEDULER_PARAMS["clip_sample"],
        timestep_spacing=SCHEDULER_PARAMS["timestep_spacing"],
        beta_schedule=SCHEDULER_PARAMS["beta_schedule"],
        steps_offset=SCHEDULER_PARAMS["steps_offset"],
    )
    pipe.scheduler = scheduler

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    return pipe, adapter

def features_to_prompts(features: dict) -> list[str]:
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
    for i, concept in enumerate(concepts):
        prompt = get_prompt('video')
        # get features!
        tempo = features["absolute"]["tempo"]
        audio_duration = features["absolute"]["audio_duration"]
        video_frame_duration = features["absolute"]["video_frame_duration"]
        average_loudness = features["absolute"]["average_loudness"]
        current_loudness = features["relative"]["rms"][i] # RMS as a proxy to loudness
        spectral_centroid = features["relative"]["spectral_centroid"][i]
        zero_crossing_rate = features["relative"]["zero_crossing_rate"][i]
        # format the prompt!
        prompt = prompt.format(
            concepts=concept,
            tempo=tempo,
            audio_duration=audio_duration,
            video_frame_duration=video_frame_duration,
            average_loudness=average_loudness,
            current_loudness=current_loudness,
            spectral_centroid=spectral_centroid,
            zero_crossing_rate=zero_crossing_rate
        )
        prompts.append(prompt)

    return prompts

def generate_videos(prompts: list[str], output_folder: str, features: dict, pipe: AnimateDiffPipeline = load_default_pipe()):
    """
    Generate images based on the music features extracted from the audio file.
    @param prompts: a list of prompts to generate, one image per prompt
    @param output_folder: the folder where the generated images will be saved
    @param pipe: the pipeline to use for generating images
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames = []
    seed = random.randint(0, 2**32)
    for i, prompt in enumerate(prompts):
        scene = pipe(
            prompt, 
            negative_prompt=MODEL_PARAMS['negative_prompt'],
            num_frames=int(MODEL_PARAMS['fps'] * features["absolute"]["video_frame_duration"]),
            guidance_scale=MODEL_PARAMS['guidance_scale'],
            num_inference_steps=MODEL_PARAMS['num_inference_steps'],
            generator=torch.Generator().manual_seed(seed),
            output_type="np" # numpy array
        )
        frames.extend(scene.frames[0])
    output_file = "./output/test_video.mp4" # TODO: make this dynamic
    # frames_pil = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    export_to_video(frames, output_file)
    return frames
        
        

