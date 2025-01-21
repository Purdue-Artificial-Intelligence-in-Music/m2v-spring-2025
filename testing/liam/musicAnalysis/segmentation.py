"""
This script includes testing code for music segmentation!
As of now, it only uses librosa to segment audio.
Feel free to test ruptures here!

Todos:
TODO 1: test ruptures

"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load audio
y, sr = librosa.load('data/racingNight.mp3', sr=44100)

# Compute a feature matrix (using Mel spectrogram)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)  # Use librosa.core.melspectrogram

# Find music segmentation boundaries
boundaries = librosa.segment.agglomerative(S, k=8)  # k is the number of segments you want to find

# Convert frame boundaries to time (seconds)
boundary_times = librosa.frames_to_time(boundaries, sr=sr)
print(type(boundary_times))

print("Segment boundaries in seconds:", boundary_times)

for t in boundary_times:
    # do stuff
    pass
