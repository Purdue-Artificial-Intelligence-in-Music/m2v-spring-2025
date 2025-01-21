x"""
This file contains various emotional analysis methods, that accept an audio file as input
and return either chords (groupings of harmonics) or "concepts" (mappings of interval changes to text descriptions).
"""

import numpy as np
import librosa
from pychord import Chord

"""
NOTE_TO_MIDI: conversions from note names to MIDI numbers within the [0,12) range, with -1 for everything undetected
INTERVAL_CONCEPTS: conversions from MIDI nubers within the [0,12) range to "concepts" for the image prompt
"""
NOTE_TO_MIDI = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11, "Unknown": -1, "Silent": -1}
INTERVAL_CONCEPTS = {
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

def map_interval_changes_to_concepts(chords: list[Chord]) -> list[str]:
    """
    Maps interval changes between chords to emotional concepts.
    @param chords: list of chords
    @return: list of emotional concepts
    """
    concepts = []
    for i in range(1, len(chords)):
        prev_midi = NOTE_TO_MIDI.get(chords[i - 1], -1)
        curr_midi = NOTE_TO_MIDI.get(chords[i], -1)
        if prev_midi == -1 or curr_midi == -1:
            concept = INTERVAL_CONCEPTS[-1]
        else:
            interval = (curr_midi - prev_midi) % 12 # IMPORTANT! represents the interval between the two notes, no matter what the notes are.
            concept = INTERVAL_CONCEPTS.get(interval, "a moment of change, an unfolding story")
        concepts.append(concept)
    
    return concepts

def chord_detection(y: np.ndarray, sr: int, num_windows: int) -> list[Chord]:
    """
    Does chord detection on a given input audio array and returns a list of chords
    @param y: the audio array
    @param sr: the sample rate
    @param num_windows: the number of windows (for sliding window feature computation)
    @return: a dictionary containing the extracted features
    """
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

    return chords
