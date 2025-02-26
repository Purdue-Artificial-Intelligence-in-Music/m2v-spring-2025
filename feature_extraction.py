"""
This file contains methods that extract audio features and concepts from an audio file.
The most important method is analyze_audio, which accepts an audio file as a path and 
returns a feature dict with the same structure as **features**.
"""

from emotion import chord_detection, map_interval_changes_to_concepts
import librosa
import numpy as np

"""
This is a description of the features dictionary and its file structure.
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
"""

def feature_dict_validity(features: dict) -> bool:
    """
    Checks if the feature dictionary is valid.
    
    @param features: The feature dictionary.
    @return: True if the feature dictionary is valid, else False.
    """
    required_relative_keys = {"rms", "spectral_centroid", "zero_crossing_rate", "concepts"}
    required_absolute_keys = {"tempo", "audio_duration", "video_frame_duration"}

    return (
        isinstance(features, dict)
        and "relative" in features
        and "absolute" in features
        and isinstance(features["relative"], dict)
        and isinstance(features["absolute"], dict)
        and required_relative_keys.issubset(features["relative"])
        and required_absolute_keys.issubset(features["absolute"])
    )


def compute_absolute_features(y: np.ndarray, 
                              sr: int, 
                              debug_print: bool = False) -> dict:
    """
    Extracts absolute features from an audio array.
    @param y: the audio array
    @param sr: the sample rate
    @param debug_print: whether to print debug information
    @return: a dictionary containing the extracted features
    """
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo.item()
    audio_duration = len(y) / sr # audio duration (in seconds)
    beats_per_measure = 4  # this is our assumption! TODO: either compute this or remove its usage.
    beat_duration = 60.0 / tempo # duration of 1 beat (in seconds)
    window_duration = beat_duration * beats_per_measure # NOTE: our window is 1 measure long, assuming 4 beats per measure. Change as needed.
    window_length = int(window_duration * sr) # length of sliding window (in samples)
    num_windows = len(y) // window_length

    ### DEBUG
    if debug_print:
        print(f"Tempo: {tempo}")
        print(f"Audio duration: {audio_duration}")
        print(f"Window duration: {window_duration}")
        print(f"Number of windows: {num_windows}")
        print(f"Window length: {window_length}")

    # Store absolute features
    abs_features = {}
    abs_features["tempo"] = tempo
    abs_features["average_loudness"] = np.mean(librosa.feature.rms(y=y))
    abs_features["audio_duration"] = audio_duration
    abs_features["video_frame_duration"] = window_duration
    abs_features["num_windows"] = num_windows
    abs_features["window_length"] = window_length

    return abs_features

def compute_relative_features(y: np.ndarray, 
                              sr: int, 
                              num_windows: int, 
                              window_length: int) -> dict:
    """
    Extracts relative features from an audio array.
    @param y: the audio array
    @param sr: the sample rate
    @param num_windows: the number of windows (for sliding window feature computation)
    @param window_length: the length of each window (in samples)
    @return: a dictionary containing the extracted features
    """
    rel_features = { 
        "rms": [], 
        "spectral_centroid": [],
        "zero_crossing_rate": [],
    }
    # Iterate via sliding window!
    for i in range(num_windows):
        start = i * window_length
        end = start + window_length
        window = y[start:end] # the window
        rms = np.mean(librosa.feature.rms(y=window))
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=window))

        # store features
        rel_features["rms"].append(rms)
        rel_features["spectral_centroid"].append(spectral_centroid)
        rel_features["zero_crossing_rate"].append(zero_crossing_rate)

    chords = chord_detection(y, sr, num_windows)
    rel_features["concepts"] = map_interval_changes_to_concepts(chords)
                                
    return rel_features

def analyze_audio(audio_file_path: str) -> dict:
    """
    Analyzes the audio file and extracts relevant features from it.
    :param file_path: path to the audio file
    :return: a dictionary containing the extracted features
    """
    # Create music features dictionary (stores both relative and absolute features)
    # Might also contain information relevant to video generation, like video frame length (will be stored in the 'absolute' dict)
    # 'features': a dictionary containing 2 dictionaries: 'relative' and 'absolute'

    # Load audio, define constants
    y, sr = librosa.load(audio_file_path, sr=None)
    features = {"relative": {}, "absolute": compute_absolute_features(y, sr)}
    features["relative"] = compute_relative_features(y, sr, features["absolute"]["num_windows"], features["absolute"]["window_length"])
    assert(feature_dict_validity(features))

    return features
