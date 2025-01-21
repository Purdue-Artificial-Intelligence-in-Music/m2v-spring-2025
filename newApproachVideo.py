"""
This script includes code for our new approach for Mus2Vid, for video generation!

Todos:
TODO 1: Implement segmentation, adapt code to use it (i.e., create video prompt per music segment)
"""

import librosa
import numpy as np
from moviepy import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips
from diffusers import StableDiffusionPipeline
import os
from PIL import Image
from pychord import Chord

# Step 0: Video generation stuff

############################### SCHEDULER PARAMETERS ###############################

schedulerParams = {
    "clip_sample": False, # Having this as True can make video become very fuzzy
    "timestep_spacing": "linspace", # "linspace", "log?"
    "beta_schedule": "linear",
    "steps_offset": 5
}

############################### LOAD VIDEO MODEL ##########################

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=schedulerParams["clip_sample"],
    timestep_spacing=schedulerParams["timestep_spacing"],
    beta_schedule=schedulerParams["beta_schedule"],
    steps_offset=schedulerParams["steps_offset"],
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

#############################################################################

# Step 1: Analyze the audio file

def analyze_audio(file_path):
    
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
                interval = (curr_midi - prev_midi) % 12
                concept = interval_concepts.get(interval, "a moment of change, an unfolding story")
            concepts.append(concept)
        
        return concepts

    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    # Create music info dictionary
    music_info = {}

    # Harmonic component extraction (removes percussive noise, isolates melody/pitches being played)
    y_harmonic = librosa.effects.harmonic(y)

    # Compute chromagram using harmonic info (to identify chords)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr) # rows: 12 pitches, this order: (C, C#, ..., B) | columns: time
    print("Time Frames:", len(chroma[0])) # number of frames

    # Estimate tempo
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    music_info["bpm"] = bpm

    # Simplistic chord detection heuristic
    chords = []
    for frame in chroma.T:  # Each row is a "frame": a few milliseconds of audio. Each column is (the loudness of) a pitch.
        # Thresholded pitches (captures the most present pitches: most likely the notes being played!)
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
    music_info["concepts"] = concepts
    return music_info

# Step 2: Map interval changes to abstract concepts


# Step 3: Generate visuals based on emotions
def generate_images(bpm, concepts, output_folder):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
    pipe.to("cuda")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = []
    for i, concept in enumerate(concepts):
        if i > 200:
            break
        prompt = f"A surreal painting representing {concept}; conveying the musical chaotic peace associated with the BPM value of {bpm}, cinematic"
        image = pipe(prompt, negative_prompt="humans, shapes, letters, low quality").images[0]
        image_name = f"frame_{i:03d}.png"
        image_path = os.path.join(output_folder, image_name)
        image.save(image_path)
        images.append(image_path)

    return images

# Step 4: Compile images into a video
def create_video(image_directory, music_info, audio_file, output_video):
    # bpm / 60 = 4 / x (4 * 60 / bpm) = x
    clips = [ImageClip(f'{image_directory}/' + img).with_duration(240 / music_info['bpm']) for img in os.listdir(image_directory)]
    video = concatenate_videoclips(clips, method="compose")

    audio = AudioFileClip(audio_file)
    video = video.with_audio(audio)

    video.write_videofile(output_video, fps=24, codec="libx264")

# Main workflow
def audio_to_emotion_video(audio_file, output_video):
    print("Analyzing audio...")
    music_info = analyze_audio(audio_file)
    bpm = music_info["bpm"]
    concepts = music_info["concepts"]
    
    print("Generating images...")
    output_folder = "generated_frames"
    images = generate_images(bpm, concepts, output_folder)

    print("Creating video...")
    create_video(images, audio_file, output_video)

    print("Video created successfully!")

# Example usage
audio_file = "data/racingNight.mp3"
output_video = "attempt2.mp4"
#analyze_audio(audio_file)
# audio_to_emotion_video(audio_file, output_video)
create_video("generated_frames", audio_file, output_video)
