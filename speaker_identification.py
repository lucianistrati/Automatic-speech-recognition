from pyAudioAnalysis import audioSegmentation
from matplotlib import pyplot as plt

import json

from src.utils import get_wav_duration


def identify_speakers_py_audio_analysis(audio_file_path):
    # Perform speaker diarization
    speakers_flags, _, _ = audioSegmentation.speaker_diarization(audio_file_path, 2)

    return str(speakers_flags.tolist())


def plot_speakers_parts(timestamp_segments_to_speaker):
    """
    Plot a horizontal bar chart representing the segmentation of the conversation by speakers.

    Parameters:
    - timestamp_segments_to_speaker (list of tuples): List of tuples containing timestamp segments
      associated with respective speakers. Each tuple has the format (start_time, end_time, speaker_name).
    """

    # Extract speaker names and corresponding start and end timestamps
    speakers = [segment[0] for segment in timestamp_segments_to_speaker]
    start_times = [segment[1] for segment in timestamp_segments_to_speaker]
    end_times = [segment[2] for segment in timestamp_segments_to_speaker]

    # Calculate durations for each speaker
    durations = [end - start for start, end in zip(start_times, end_times)]

    # Create a horizontal bar chart
    plt.barh(speakers, durations, color='skyblue')

    # Set labels and title
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Speakers')
    plt.title('Speaker Segmentation Over Time')
    plt.savefig("images/speaker_segmentation_over_time_intervals.png")
    # Display the plot
    plt.show()


def get_speaker_segments(sequence, frame_duration=0.1):
    # Assuming frame_duration is the duration of each frame in seconds
    # Adjust as needed based on your specific audio file characteristics

    segments = []
    current_segment = {"speaker": None, "start_time": None}

    for i, value in enumerate(sequence):
        if current_segment["speaker"] is None:
            # Start of a new segment
            current_segment["speaker"] = value
            current_segment["start_time"] = i * frame_duration
        elif current_segment["speaker"] != value:
            # Transition to a new speaker, end the current segment
            current_segment["end_time"] = i * frame_duration
            segments.append(current_segment.copy())
            current_segment["speaker"] = value
            current_segment["start_time"] = i * frame_duration

    # Handle the last segment
    if current_segment["speaker"] is not None:
        current_segment["end_time"] = len(sequence) * frame_duration
        segments.append(current_segment)

    return segments


def speaker_identification(audio_file_path: str = '../data/short_audio_2_speakers.wav'):
    print("#" * 100)
    print("speaker_identification")

    transcript_with_speakers = ""
    speakers_flags = identify_speakers_py_audio_analysis(
        audio_file_path)

    with open("../data/transcript_with_speakers_short_py_audio_analysis.txt", "w") as f:
        f.write(speakers_flags)
    duration = get_wav_duration("../data/short_audio_2_speakers.wav")
    (transcript_text, text_turns, timestamp_to_content, turns_times, flags,
     confidence_scores) \
        = (
        extract_turns(
            "../data/transcript_short_2_speakers_aws_segmented.json",
            duration))
    (transcript_text, text_turns, timestamp_to_content, turns_times, flags,
     confidence_scores) \
        = (
        extract_turns(
            "../data/transcript_aws_segmented.json",
            duration))
    print(timestamp_to_content)
    print(turns_times)


if __name__ == "__main__":
    speaker_identification()
