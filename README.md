# Mus2Vid: The Pivot!

This is the Github project containing all code pertaining to Mus2Vid's pivot, starting in Spring 2025.

## How to run

1. Make a new Conda environment with Python 3.12.7, and activate said environment.
(If on Gilbreth (one of Purdue's clusters), refer to https://www.rcac.purdue.edu/knowledge/gilbreth/run/examples/apps/python/conda)
2. Clone this repo and ``cd`` to where the files are downloaded.
3. Run ``pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0``
4. Run ``conda install -c conda-forge vapoursynth`` (the 'vsrife' module needs VapourSynth, which is troublesome to install normally)
5. Run ``pip install -r requirements.txt``.
6. Put your input audio files in ``./input``. ".wav" or ".mp3" files only, please.
7. Create your own sinteractive script in the 'scripts' directory by copy-pasting the 'sinteractive' file and replacing paths with your own.
8. Run all commands in your sinteractive script in your terminal. (you can copy-paste for ease)
9. Done! This should automatically run ``main.py``, the file that actually runs the program.

## Adding your own LLM API Keys

1. Create a text file in the 'keys' directory with the following name format: "{type}ApiKey.txt". {type} must be lowercase.
    - examples: "gemeniApiKey.txt", "gptApiKey.txt"
2. Use the 'get_api_key(type)' method in utils.py!

## Important Format to Follow

You'll notice that most scripts have a multi-line comment at the top, explaining what the script is about, its purpose, etc. 

Please do the same for files you create, it will help us all out. By the same token, please comment your code so that others know what different lines do.
