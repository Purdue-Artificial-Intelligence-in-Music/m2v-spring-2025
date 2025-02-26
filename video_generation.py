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
from llm import gemini
from PIL import Image
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video
import cv2
from vsrife import rife
import numpy as np
import vapoursynth as vs
from util import get_prompt
from moviepy import AudioFileClip, ImageSequenceClip, ImageClip, concatenate_videoclips


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
    "num_inference_steps": 5,
    "fps": 8 # I don't think we can change this. 8 is the default for AnimateDiff
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
    return pipe

def features_to_llm_prompts(features: dict) -> list[str]:
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
            type='video',
            frame_description="sequence of evolving visuals",
            motion_fast="Scenes with rapid motion, dramatic camera sweeps, intense lighting shifts.",
            motion_slow="Still, dreamlike, atmospheric movements, focusing on soft transitions.",
            motion_mid="Balanced motion with gradual, smooth evolution.",
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


def llm_to_video_prompts(llm_prompts: list[str]):
    video_prompts = []
    print(f"Prompting the LLM with {len(llm_prompts)} scene prompts...")
    # Now prompt the LLM!
    for prompt in llm_prompts:
        video_prompt = gemini(prompt)
        video_prompts.append(video_prompt)
    return video_prompts

# TODO: fix bug where video is black screen with NO audio
def generate_videos(prompts: list[str], output_folder: str, features: dict, audio_file: str, pipe: AnimateDiffPipeline = load_default_pipe()):
    """
    Generate a video from AnimateDiff frames without saving images and synchronize it with an audio file.

    @param prompts: List of prompts for generating scenes.
    @param output_folder: Directory where the final video will be saved.
    @param features: Dictionary of extracted audio features.
    @param audio_file: Path to the MP3 file.
    @param pipe: AnimateDiff pipeline for generating frames.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get audio duration
    audio = AudioFileClip(audio_file)
    audio_duration = audio.duration  # in seconds

    fps = MODEL_PARAMS['fps']
    total_frames = int(audio_duration * fps)  # Ensure video matches audio length
    num_frames_per_scene = max(4, total_frames // len(prompts))  # Distribute frames per scene

    print(f"Generating {len(prompts)} scenes, total frames: {total_frames}...")

    frames = []
    seed = random.randint(0, 2**32)
    for i, prompt in enumerate(prompts):
        scene = pipe(
            prompt, 
            negative_prompt=MODEL_PARAMS['negative_prompt'],
            num_frames=num_frames_per_scene,
            guidance_scale=MODEL_PARAMS['guidance_scale'],
            num_inference_steps=MODEL_PARAMS['num_inference_steps'],
            generator=torch.Generator().manual_seed(seed),
            output_type="np" # numpy array
        )
        frames.extend(scene.frames[0])
    
    # Ensure frame count matches expected total frames
    if len(frames) < total_frames:
        last_frame = frames[-1]
        frames.extend([last_frame] * (total_frames - len(frames)))
    elif len(frames) > total_frames:
        frames = frames[:total_frames]

    # Convert NumPy frames to ImageSequenceClip directly (No file I/O)
    video = ImageSequenceClip(frames, fps=fps)
    video = video.with_audio(audio)

    # Set the output file name
    i = 1
    output_file = f"{output_folder}/test_video_{i}.mp4" # TODO: make this dynamic
    while os.path.isfile(output_file):
        i += 1
        output_file = f"./output/test_video_{i}.mp4"


    video.write_videofile(output_file, fps=fps, codec="libx264", audio_codec="aac")
    print(f"Final video saved: {output_file}")
    # return output_file

        
        

