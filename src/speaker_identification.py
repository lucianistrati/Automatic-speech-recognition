from pyAudioAnalysis import audioSegmentation
from matplotlib import pyplot as plt
from typing import List, Tuple, Dict, Any
import json


def identify_speakers_py_audio_analysis(audio_file_path: str, num_speakers: int = 2) -> List[int]:
    """Perform speaker diarization using the pyAudioAnalysis library.

    Args:
        audio_file_path (str): Path to the audio file.
        num_speakers (int): The number of speakers to identify.

    Returns:
        List[int]: A list of integers indicating speaker labels for each frame.
    """
    try:
        speakers_flags, _, _ = audioSegmentation.speaker_diarization(audio_file_path, num_speakers)
        return speakers_flags.tolist()
    except Exception as e:
        raise Exception(f"Error during speaker diarization: {e}")


def plot_speakers_parts(timestamp_segments_to_speaker: List[Tuple[float, float, str]]):
    """Plot a horizontal bar chart representing the segmentation of the conversation by speakers.

    Args:
        timestamp_segments_to_speaker (List[Tuple[float, float, str]]): List of tuples containing 
        timestamp segments associated with respective speakers. 
        Each tuple has the format (start_time, end_time, speaker_name).
    """
    # Extract speaker names and corresponding start and end timestamps
    speakers = [segment[2] for segment in timestamp_segments_to_speaker]
    start_times = [segment[0] for segment in timestamp_segments_to_speaker]
    end_times = [segment[1] for segment in timestamp_segments_to_speaker]

    # Calculate durations for each speaker
    durations = [end - start for start, end in zip(start_times, end_times)]

    # Create a horizontal bar chart
    plt.barh(speakers, durations, color='skyblue')

    # Set labels and title
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Speakers')
    plt.title('Speaker Segmentation Over Time')
    plt.savefig("images/speaker_segmentation_over_time_intervals.png")
    plt.show()


def get_speaker_segments(sequence: List[int], frame_duration: float = 0.1) -> List[Dict[str, Any]]:
    """Extract speaker segments from the diarization sequence.

    Args:
        sequence (List[int]): The diarization sequence indicating speaker changes.
        frame_duration (float): Duration of each frame in seconds.

    Returns:
        List[Dict[str, Any]]: A list of segments where each segment contains speaker info and timing.
    """
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
    """Main function to identify speakers in the audio file.

    Args:
        audio_file_path (str): Path to the audio file for speaker identification.
    """
    print("#" * 100)
    print("speaker_identification")

    try:
        # Identify speakers and save results
        speakers_flags = identify_speakers_py_audio_analysis(audio_file_path)
        with open("../data/transcript_with_speakers_short_py_audio_analysis.txt", "w") as f:
            f.write(json.dumps(speakers_flags))

        # Get audio duration
        duration = get_wav_duration(audio_file_path)

        # Extract transcript turns
        transcript_info = extract_turns("../data/transcript_short_2_speakers_aws_segmented.json", duration)
        transcript_text, text_turns, timestamp_to_content, turns_times, flags, confidence_scores = transcript_info

        print(timestamp_to_content)
        print(turns_times)
    except Exception as e:
        print("Error in speaker identification process:", e)


if __name__ == "__main__":
    speaker_identification()
