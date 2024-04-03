# Automatic speech recognition

## Install dependencies

Install all the project dependencies by running the following command in the 
project's root directory:

```bash
poetry install
```

You can use the following command to activate the virtual environment

```bash
poetry shell
```

Note: Make sure you have Poetry installed on your system. If not, you can install it
using:

```bash
pip install poetry
```

## Other prerequisites

Ensure you have the following API key before stored in the .env file:

- OPEN_AI_API_KEY

## Run the code

After completing those steps you can add a youtube url of a video in the main.py 
file and run that script, this should tackle the download, conversion to wav file and 
some other visualizations which woudld be obtained afterwards:

```bash
python src/main.py
```

## Documentation

For more details of the implementation you may check "documentation/documentation.MD" 
file

## Issues

### OPEN AI issue

If you encounter any issues with the open ai version, run this:

```bash
pip install openai==0.28
```

### Path issue

If you encounter issues with the relative path, run this:

```bash
export PYTHONPATH=$PWD
```
