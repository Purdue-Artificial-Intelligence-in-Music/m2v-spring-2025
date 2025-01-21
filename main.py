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
from image_generation import generate_images
from util import create_video, clean_up

def audio_to_emotion_video(audio_file: str, 
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
    images = generate_images(features, output_folder)

    # Step 3: Create video
    if debug_print:
        print("Creating video...")
    create_video(features, output_folder, audio_file, output_video_path)
    if debug_print:
        print("Video created successfully!")

    # Step 4: Clean up
    if debug_print:
        print("Cleaning up...")
    clean_up(output_folder)
    if debug_print:
        print("Done!")

if __name__ == "__main__":
    audio_file = "input/playingGodUkulele.mp3"
    output_video = "output/playingGod1.mp4"
    audio_to_emotion_video(audio_file, output_video, debug_print=True)
