"""
This script includes code for our new approach for Mus2Vid!
This script only generates images, and simply provides a base proof of concept.
I (Liam) am working on converting this code into something that generates videos as well.
Soon, this process will be delegated in different parts to the rest of the team.

Todos:
TODO 1: Implement segmentation, adapt code to use it (in 'newApproachVideo.py')
TODO 2: Remove reliance on the '4 beats per measure' assumption

Notes:
NOTE 1: The time duration of a single beat is (60 / bpm)
NOTE 2: The difference between an 'audio frame' and a 'video frame':
    - an audio frame is a small sample of audio data, a few milliseconds long
    - a video frame is a single image
NOTE 3: Tempo and BPM (Beats Per Minute) are the same thing. 
"""

from feature_extraction import analyze_audio
from image_generation import generate_images, features_to_image_prompts
from video_generation import generate_videos, features_to_llm_prompts, llm_to_video_prompts
from util import frames_to_video, clean_up
from diffusers.utils import logging
import os
import warnings

# Disable warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="You have disabled the safety checker")


INPUT_DIR = "./input"
OUTPUT_DIR = "./output"

def video_pipe(audio_file: str, 
                           output_video_path: str, 
                           debug_print: bool = False):
    """
    Generate a video based on the features of the audio file.
    @param audio_file: the path to the audio file
    @param output_video: the path to save the output video
    """

    # Step 1: Extract music information
    if debug_print:
        print("Analyzing audio...")
    features = analyze_audio(audio_file)
    
    # Step 2: Generate LLM and video prompts
    output_folder = "./output"
    if debug_print:
        print("Generating LLM prompts...")
    llm_prompts = features_to_llm_prompts(features)
    if debug_print:
        print("Generating Video Generation prompts...")
    video_prompts = llm_to_video_prompts(llm_prompts)

    # Step 3: Create video
    if debug_print:
        print("Creating video...")
    generate_videos(video_prompts, output_folder, features, audio_file)
    if debug_print:
        print("Video created successfully!")
    if debug_print:
        print("Done!")

# Our image generation pipeline.
def image_pipe(audio_file: str, 
                           output_video_path: str, 
                           debug_print: bool = False):
    """
    Generate a video based on the features of the audio file.
    @param audio_file: the path to the audio file
    @param output_video: the path to save the output video
    """

    # Step 1: Extract music information
    if debug_print:
        print("Analyzing audio...")
    features = analyze_audio(audio_file)
    
    # Step 2: Generate frames
    if debug_print:
        print("Generating images...")
    output_folder = "generated_frames"
    prompts = features_to_image_prompts(features)
    generate_images(prompts, output_folder)

    # Step 3: Create video
    if debug_print:
        print("Creating video...")
    frames_to_video(features, output_folder, audio_file, output_video_path)
    if debug_print:
        print("Video created successfully!")

    # Step 4: Clean up
    if debug_print:
        print("Cleaning up...")
    clean_up(output_folder)
    if debug_print:
        print("Done!")

def main():
    # Select pipeline
    pipe_type = input("Type 'video' or 'image' to select a pipeline: ")
    pipe = video_pipe if pipe_type == "video" else image_pipe

    # Get input file
    default_file = "playingGodUkulele.mp3"
    input_file = input("Enter input file name WITH file extension (must be inside 'input' directory),\
        or ENTER for default test file: ")
    if input_file == "": input_file = default_file
    input_path = f"{INPUT_DIR}/{input_file}"
    while os.path.isdir(input_path):
        print("That is a directory. Please enter a file instead.")
        input_file = input("Enter input file (must be inside 'input' directory): ")
        input_path = f"{INPUT_DIR}/{input_file}"

    # Process!
    print("Processing", input_file)
    output_video = os.path.join(OUTPUT_DIR, input_file[:-4] + ".mp4")
    pipe(input_path, output_video, debug_print=True)


if __name__ == "__main__":
    main()