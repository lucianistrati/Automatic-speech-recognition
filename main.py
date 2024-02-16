from pytube import YouTube

import json
from typing import List, Tuple
from collections import Counter


def download_video(url: str) -> str:
    saving_path = YouTube(url).streams.first(

    ).download()
    return saving_path


download_video("")
exit(0)
from automatic_speech_recognition import automatic_speech_recognition
from speaker_identification import speaker_identification
from converters import convert
from transcript_analysis import transcript_analysis
from transcript_quality_improvement_comparison import\
    transcript_quality_improvement_comparison
from moviepy.editor import VideoFileClip
from pydub import AudioSegment


def convert_mp4_to_wav(input_mp4, output_wav):
    video_clip = VideoFileClip(input_mp4)
    audio_clip = video_clip.audio

    # Save the audio clip as a WAV file
    audio_clip.write_audiofile(output_wav, codec='pcm_s16le', ffmpeg_params=['-ac', '1'])


def convert_mp3_to_wav(mp3_file, wav_file):
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_file)

    # Export the audio to WAV format
    audio.export(wav_file, format="wav")


def main():
    url = "https://www.youtube.com/watch?v=iHibnmosKkM"
    saving_path = download_video(url)
    if saving_path.endswith(".mp3"):
        convert_mp3_to_wav(saving_path, "example.wav")
    elif saving_path.endswith(".mp4"):
        convert_mp4_to_wav(saving_path, "example.wav")
    elif saving_path.endswith(".wav"):
        pass
    else:
        raise ValueError("Unknown file extension!")
    automatic_speech_recognition()
    speaker_identification()
    transcript_analysis()
    transcript_quality_improvement_comparison()


if __name__ == "__main__":
    main()