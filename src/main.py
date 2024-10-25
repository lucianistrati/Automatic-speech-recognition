import json
from pytube import YouTube
from typing import Optional
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from automatic_speech_recognition import automatic_speech_recognition
from speaker_identification import speaker_identification
from converters import convert
from transcript_analysis_2 import transcript_analysis
from transcript_quality_improvement_comparison import transcript_quality_improvement_comparison


def download_video(url: str) -> str:
    """Download a YouTube video and return the file path.

    Args:
        url (str): The URL of the YouTube video to download.

    Returns:
        str: The path to the downloaded video file.
    """
    try:
        video = YouTube(url)
        saving_path = video.streams.first().download()
        return saving_path
    except Exception as e:
        raise Exception(f"Failed to download video from {url}: {e}")


def convert_mp4_to_wav(input_mp4: str, output_wav: str):
    """Convert an MP4 video file to a WAV audio file.

    Args:
        input_mp4 (str): Path to the input MP4 file.
        output_wav (str): Path to save the output WAV file.
    """
    try:
        video_clip = VideoFileClip(input_mp4)
        audio_clip = video_clip.audio

        # Save the audio clip as a WAV file
        audio_clip.write_audiofile(output_wav, codec='pcm_s16le', ffmpeg_params=['-ac', '1'])
    except Exception as e:
        raise Exception(f"Error converting {input_mp4} to {output_wav}: {e}")


def convert_mp3_to_wav(mp3_file: str, wav_file: str):
    """Convert an MP3 audio file to a WAV audio file.

    Args:
        mp3_file (str): Path to the input MP3 file.
        wav_file (str): Path to save the output WAV file.
    """
    try:
        # Load the MP3 file
        audio = AudioSegment.from_mp3(mp3_file)

        # Export the audio to WAV format
        audio.export(wav_file, format="wav")
    except Exception as e:
        raise Exception(f"Error converting {mp3_file} to {wav_file}: {e}")


def main():
    """Main function to execute the workflow of downloading a video,
    converting it to WAV, and performing analyses.
    """
    url = "https://www.youtube.com/watch?v=iHibnmosKkM"

    # Download the video from YouTube
    saving_path = download_video(url)

    # Determine file type and convert to WAV if necessary
    if saving_path.endswith(".mp3"):
        convert_mp3_to_wav(saving_path, "example.wav")
    elif saving_path.endswith(".mp4"):
        convert_mp4_to_wav(saving_path, "example.wav")
    elif saving_path.endswith(".wav"):
        print("The downloaded file is already in WAV format.")
    else:
        raise ValueError("Unknown file extension!")

    # Perform analyses
    automatic_speech_recognition("example.wav")
    speaker_identification("example.wav")
    transcript_analysis("example.wav")
    transcript_quality_improvement_comparison("example.wav")


if __name__ == "__main__":
    main()
