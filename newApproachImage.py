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

import librosa
import numpy as np
from moviepy import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips
from diffusers import StableDiffusionPipeline
import os
import shutil
from PIL import Image
from pychord import Chord

# Step 1: Analyze the audio file
def analyze_audio(file_path):
    ##############################################################################################
    ##################################### HELPER METHODS ##########################################

    # TODO: make this better
    def map_interval_changes_to_concepts(chords):
        note_to_midi = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11, "Unknown": -1, "Silent": -1}
        interval_concepts = {
            0: "a sense of stability and reflection",
            1: "a gentle shift, a moment of curiosity",
            2: "a rising anticipation, an evolving journey",
            3: "a melancholy shift, a fading memory",
            4: "a hopeful ascent, a moment of clarity",
            5: "a major transition, a bold step forward",
            6: "a subtle tension, an unresolved feeling",
            7: "a triumphant leap, an ambitious endeavor",
            8: "a mysterious turn, an unexpected twist",
            9: "a dramatic change, an emotional surge",
            10: "a gentle return, a nostalgic feeling",
            11: "a hesitant step, an uncertain future",
            -1: "a pause, an uncertain moment",
        }

        concepts = []
        for i in range(1, len(chords)):
            prev_midi = note_to_midi.get(chords[i - 1], -1)
            curr_midi = note_to_midi.get(chords[i], -1)
            if prev_midi == -1 or curr_midi == -1:
                concept = interval_concepts[-1]
            else:
                interval = (curr_midi - prev_midi) % 12 # IMPORTANT! represents the interval between the two notes, no matter what the notes are.
                concept = interval_concepts.get(interval, "a moment of change, an unfolding story")
            concepts.append(concept)
        
        return concepts

    ##############################################################################################
    ##################################### ANALYZE AUDIO ##########################################

    # Create music features dictionary (stores both relative and absolute features)
    # Might also contain information relevant to video generation, like video frame length (will be stored in the 'absolute' dict)
    # 'features': a dictionary containing 2 dictionaries: 'relative' and 'absolute'
    features = {"relative": {
                    "rms": [],
                    "spectral_centroid": [],
                    "zero_crossing_rate": [],
                    "concepts": []
                    }, 
                "absolute": {   # dictionary of values
                    "tempo": None,
                    "audioDuration": None,
                    "videoFrameDuration": None
                    }
                }

    # Load audio, define constants
    y, sr = librosa.load(file_path, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo.item()
    audio_duration = len(y) / sr # audio duration (in seconds)
    beats_per_measure = 4  # this is our assumption! TODO: either compute this or remove its usage.
    beat_duration = 60.0 / tempo # duration of 1 beat (in seconds)
    window_duration = beat_duration * beats_per_measure # NOTE: our window is 1 measure long, assuming 4 beats per measure. Change as needed.
    window_length = int(window_duration * sr) # length of sliding window (in samples)
    num_windows = len(y) // window_length

    ### DEBUG
    print(f"Tempo: {tempo}")
    print(f"Audio duration: {audio_duration}")
    print(f"Window duration: {window_duration}")
    print(f"Number of windows: {num_windows}")

    #####################################################
    ##########// Absolute Feature Extraction \\##########

    features["absolute"]["tempo"] = tempo
    features["absolute"]["audioDuration"] = audio_duration
    features["absolute"]["videoFrameDuration"] = window_duration

    #####################################################
    ##########// Relative Feature Extraction \\##########

    # Iterate via sliding window!
    for i in range(num_windows):
        start = i * window_length
        end = start + window_length
        window = y[start:end] # the window
        rms = np.mean(librosa.feature.rms(y=window))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=window))

        # store features
        features["relative"]["rms"].append(rms)
        features["relative"]["spectral_centroid"].append(spectral_centroid)
        features["relative"]["zero_crossing_rate"].append(zero_crossing_rate)

    ####################################################
    ##########// Chord progression analysis \\##########
    
    # Harmonic component extraction (removes percussive noise, isolates melody/pitches being played)
    y_harmonic = librosa.effects.harmonic(y)

    # Compute chromagram using harmonic info (to identify chords)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr) # rows: 12 pitches, this order: (C, C#, ..., B) | columns: time
    print("Time Frames:", len(chroma[0])) # number of frames

    # Simplistic chord detection
    chords = []
    for i in range(0, num_windows):  # Each row is a "frame": a few milliseconds of audio. Each column is (the loudness of) a pitch.
        # Thresholded pitches (captures the most present pitches: most likely the notes being played!)
        index = int(i * (len(chroma.T) / num_windows))
        frame = chroma.T[index]
        threshold = 0.5
        pitches = np.where(frame > threshold)[0] # returns a 1-lengthed tuple, so we access at [0] to get indices where frame > threshold.
        if len(pitches) > 0: # there are actually some notes being played
            try:
                notes = [librosa.midi_to_note(p + 60) for p in pitches] # NOTE: this will ALWAYS be mapped to the C4 octave. (C4 to B4)
                chord = Chord(notes[0]) if notes else "Unknown"
                chords.append(chord.root)
            except ValueError:
                chords.append("Unknown")
        else:
            chords.append("Silent")

    # extract chord progressions/changes
    concepts = map_interval_changes_to_concepts(chords)
    features["relative"]["concepts"] = concepts

    return features


# Step 3: Generate visuals based on emotions
def generate_images(features, output_folder):
    # Unpack music info
    tempo = features["absolute"]["tempo"]
    concepts = features["relative"]["concepts"]

    # Generate images
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
    pipe.to("cuda")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = []
    for i, concept in enumerate(concepts):
        if i > 200:
            break
        prompt = f"A surreal painting representing {concept}; conveying the musical chaotic peace associated with the Tempo value of {tempo}; cinematic!"
        image = pipe(prompt, negative_prompt="humans, shapes, letters, low quality, blurry, painting frame").images[0]
        image_name = f"frame_{i:03d}.png"
        image_path = os.path.join(output_folder, image_name)
        image.save(image_path)
        

# Step 4: Compile images into a video
def create_video(features, image_directory, audio_file, output_video_path):
    # unpack relevant features
    videoFrameDuration = 1.75 if features is None else features["absolute"]["videoFrameDuration"] 

    # create video
    clips = [ImageClip(f'{image_directory}/' + img).with_duration(videoFrameDuration) for img in os.listdir(image_directory)]
    video = concatenate_videoclips(clips, method="compose")

    audio = AudioFileClip(audio_file)
    video = video.with_audio(audio)

    video.write_videofile(output_video_path, fps=24, codec="libx264")

# Putting it all together!
def audio_to_emotion_video(audio_file, output_video):
    # Step 1: Extract music information
    print("Analyzing audio...")
    features = analyze_audio(audio_file)
    
    # Step 2: Generate frames
    print("Generating images...")
    output_folder = "generated_frames"
    images = generate_images(features, output_folder)

    # Step 3: Create video
    print("Creating video...")
    create_video(features, output_folder, audio_file, output_video)
    print("Video created successfully!")

    # Step 4: Clean up
    print("Cleaning up...")
    shutil.rmtree(output_folder) # delete all generated images
    print("Done!")

# Example usage
audio_file = "input/playingGodUkulele.mp3"
output_video = "output/playingGod1.mp4"
audio_to_emotion_video(audio_file, output_video)
