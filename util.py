"""
This file contains some basic utilities to process input/output and manage folders.
"""

from moviepy import AudioFileClip, ImageClip, concatenate_videoclips
import os
import shutil

def get_api_key(type="gemini"):
    with open(f'./keys/{type}ApiKey.txt', 'r') as file:
        key = file.readline()
        key = key.replace("\n", "")
        return key # READS THE FIRST LINE ONLY

def get_prompt(type="video"):
    """
    Returns the prompt specified by 'type' ("video" or "image", based on our two pipelines).
    """
    with open(f'./prompts/{type}Prompt.txt', 'r') as file:
        prompt = ' '.join(file.readlines())
        return prompt

def frames_to_video(features: dict, 
                 image_directory: str, 
                 audio_file_path: str, 
                 output_video_path: str):
    """
    Create a video from a set of images and an audio file.
    @param features: a dictionary containing the extracted features
    @param image_directory: the directory containing the images
    @param audio_file: the path to the audio file
    @param output_video_path: the path to save the output video
    """
    # unpack relevant features
    videoFrameDuration = 1.75 if features is None else features["absolute"]["video_frame_duration"] 

    # create video
    clips = [ImageClip(f'{image_directory}/' + img).with_duration(videoFrameDuration) for img in os.listdir(image_directory)]
    video = concatenate_videoclips(clips, method="compose")

    audio = AudioFileClip(audio_file_path)
    video = video.with_audio(audio)

    video.write_videofile(output_video_path, fps=24, codec="libx264")

def clean_up(output_folder: str):
    """
    Clean up the output folder.
    @param output_folder: the folder to clean up
    """
    shutil.rmtree(output_folder)