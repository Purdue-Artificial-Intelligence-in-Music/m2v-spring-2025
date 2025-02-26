# Mus2Vid: The Pivot!

This is the Github project containing all code pertaining to Mus2Vid's pivot, starting in Spring 2025.

## How to run

1. Make a new Conda environment with Python 3.12.7, and activate said environment.
(If on Gilbreth (one of Purdue's clusters), refer to https://www.rcac.purdue.edu/knowledge/gilbreth/run/examples/apps/python/conda)
2. Clone this repo and ``cd`` to where the files are downloaded.
3. Run ``pip install -r requirements.txt``.
4. Put your input audio files in ``./input``. ".wav" or ".mp3" files only, please.
5. Run ``./main.py``. Press Enter to use the default directories for input and output.
6. ``./main.py`` will generate "slideshow" videos from your music inputs.

## Adding your own LLM API Keys

1. Create a text file in the 'keys' directory with the following name format: "{type}ApiKey.txt". {type} must be lowercase.
    - examples: "gemeniApiKey.txt", "gptApiKey.txt"
2. Use the 'get_api_key(type)' method in utils.py!

## Important Format to Follow

You'll notice that most scripts have a multi-line comment at the top, explaining what the script is about, its purpose, etc. 

Please do the same for files you create, it will help us all out. By the same token, please comment your code so that others know what different lines do.
