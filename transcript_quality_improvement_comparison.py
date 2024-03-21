import json

from pydub import AudioSegment
# import language_tool_python
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


def noise_reduction(input_file, output_file):
    rate, data = wavfile.read(input_file)
    reduced_noise = reduce_noise(y=data, sr=rate)
    wavfile.write(output_file, rate, reduced_noise)


def normalize_audio(audio_file_path, output_path):
    sound = AudioSegment.from_wav(audio_file_path)
    sound = sound - sound.dBFS  # Normalize audio
    sound.export(output_path, format="wav")


def correct_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    corrected_text = tool.correct(text)
    return corrected_text


def spell_checking(text):
    spell = Speller(lang='en')
    corrected_text = " ".join([spell(word) for word in word_tokenize(text)])
    return corrected_text


def coherence_enhancement(text):
    enhanced_text = summarizer.summarize(text, ratio=0.25)
    return enhanced_text


def grammar_correction(transcript: str) -> str:
    """
    Correct grammar in a transcript.

    Parameters:
    - transcript (str): Input transcript.

    Returns:
    - str: Transcript with corrected grammar.
    """
    tool = language_tool_python.LanguageTool('en-US')

    # Check grammar and get suggestions
    matches = tool.check(transcript)

    # Apply corrections
    corrected_transcript = language_tool_python.correct(transcript, matches)

    return corrected_transcript


def transcript_quality_improvement(transcript_file_path: str):
    with open(transcript_file_path, "r") as f:
        transcript_text = f.read()

    corrected_text = spell_checking(transcript_text)
    print("Spell-checked Text:", corrected_text)

    enhanced_text = coherence_enhancement(transcript_text)
    print("Enhanced Coherence:", enhanced_text)

    # corrected_text = correct_grammar(transcript_text)
    # print("Corrected Grammar:", corrected_text)


def transcript_quality_improvement_comparison():
    print("#" * 100)
    print("transcript_quality_improvement_comparison")
    audio_file_path = "../data/audio.wav"
    output_path = '../data/cleaned_audio.wav'
    normalize_audio(audio_file_path, output_path)

    transcript_file_paths = ["data/transcript_google.txt",
                             "data/transcript_azure.txt",
                             "data/transcript_deepspeech.txt",
                             "data/transcript_google.txt"]
    for transcript_file_path in transcript_file_paths:
        transcript_quality_improvement(transcript_file_path)


if __name__ == "__main__":
    transcript_quality_improvement_comparison()
