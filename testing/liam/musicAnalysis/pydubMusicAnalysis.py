"""
This script includes testing code for general music analysis using PyDub!
PyDub might provide us additional tools when it comes to feature extraction.
All testing with PyDub can be done here!

Todos:
TODO 1: test ruptures
"""

from pydub import AudioSegment
from pydub.utils import make_chunks

# Load the MP3 file
mp3_file_path = "data/racingNight.mp3"  # Replace with your MP3 file path
audio = AudioSegment.from_file(mp3_file_path, format="mp3")
print(audio)
# Set parameters for sliding window
window_size = 5000  # 5 seconds in milliseconds
time_increment = 1000  # 1 second in milliseconds

def analyze_segment(segment, start_time):
    """Custom analysis function for each 5-second segment."""
    # Example: Get the average volume (RMS)
    rms = segment.rms
    print(f"Segment starting at {start_time / 1000:.1f}s: RMS={rms}")

# Iterate through the audio using the sliding window
start_time = 0
while start_time + window_size <= len(audio):
    segment = audio[start_time:start_time + window_size]
    # analyze_segment(segment, start_time)
    start_time += time_increment

# Handle the last segment if it is shorter than the window size
if start_time < len(audio):
    print("\n\nLAST!\n\n")
    segment = audio[start_time:]
    analyze_segment(segment, start_time)
