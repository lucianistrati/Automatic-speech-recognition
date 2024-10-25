import json
from pydub import AudioSegment
from summa import summarizer
from autocorrect import Speller
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
from typing import List
import language_tool_python
from statistics import mean, median, mode
from gensim import corpora, models
from typing import Dict
from noisereduce import reduce_noise
from scipy.io import wavfile


def reduce_noise_from_audio(input_file: str, output_file: str) -> None:
    """Reduce noise from an audio file and save the cleaned audio.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path where the cleaned audio file will be saved.
    """
    rate, data = wavfile.read(input_file)
    reduced_noise = reduce_noise(y=data, sr=rate)
    wavfile.write(output_file, rate, reduced_noise)


def normalize_audio_volume(audio_file_path: str, output_path: str) -> None:
    """Normalize the volume of a WAV audio file.

    Args:
        audio_file_path (str): Path to the original audio file.
        output_path (str): Path where the normalized audio file will be saved.
    """
    sound = AudioSegment.from_wav(audio_file_path)
    sound = sound - sound.dBFS  # Normalize audio
    sound.export(output_path, format="wav")


def correct_text_grammar(text: str) -> str:
    """Correct grammar in the provided text using LanguageTool.

    Args:
        text (str): Input text to be corrected.

    Returns:
        str: Text with corrected grammar.
    """
    tool = language_tool_python.LanguageTool('en-US')
    corrected_text = tool.correct(text)
    return corrected_text


def perform_spell_check(text: str) -> str:
    """Check and correct spelling in the provided text.

    Args:
        text (str): Input text for spell checking.

    Returns:
        str: Text with corrected spelling.
    """
    spell = Speller(lang='en')
    corrected_text = " ".join([spell(word) for word in word_tokenize(text)])
    return corrected_text


def enhance_coherence(text: str) -> str:
    """Enhance the coherence of the provided text using TextRank summarization.

    Args:
        text (str): Input text to be enhanced.

    Returns:
        str: Text with enhanced coherence.
    """
    enhanced_text = summarizer.summarize(text, ratio=0.25)
    return enhanced_text


def improve_transcript_quality(transcript_file_path: str) -> None:
    """Read, spell-check, and enhance the coherence of a transcript.

    Args:
        transcript_file_path (str): Path to the transcript file to be improved.
    """
    with open(transcript_file_path, "r") as file:
        transcript_text = file.read()

    corrected_text = perform_spell_check(transcript_text)
    print("Spell-checked Text:", corrected_text)

    enhanced_text = enhance_coherence(transcript_text)
    print("Enhanced Coherence:", enhanced_text)


def compare_transcript_quality_improvement() -> None:
    """Compare the quality of multiple transcripts and normalize the associated audio."""
    print("#" * 100)
    print("Transcript Quality Improvement Comparison")
    
    audio_file_path = "../data/audio.wav"
    output_path = '../data/cleaned_audio.wav'
    normalize_audio_volume(audio_file_path, output_path)

    transcript_file_paths = [
        "data/transcript_google.txt",
        "data/transcript_azure.txt",
        "data/transcript_deepspeech.txt",
        "data/transcript_google.txt"  # Duplicated file; consider updating if needed
    ]
    
    for transcript_file_path in transcript_file_paths:
        improve_transcript_quality(transcript_file_path)


if __name__ == "__main__":
    compare_transcript_quality_improvement()
