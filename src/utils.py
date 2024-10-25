import wave
import json
from typing import List, Tuple, Dict
from collections import Counter

def decide_intervals(content_start: float, content_end: float, speaker_start: float, speaker_end: float) -> float:
    """Determine the relationship between content and speaker time intervals.

    Args:
        content_start (float): Start time of the content.
        content_end (float): End time of the content.
        speaker_start (float): Start time of the speaker's segment.
        speaker_end (float): End time of the speaker's segment.

    Returns:
        float: Length of the overlap between intervals, or a message indicating their relationship.

    Raises:
        ValueError: If the intervals are invalid.
    """
    if content_end < speaker_start or content_start > speaker_end:
        return "no overlap"
    elif speaker_start <= content_start <= content_end <= speaker_end:
        return "contained"
    elif content_start <= speaker_start <= content_end <= speaker_end:
        return content_end - speaker_start
    elif speaker_start <= content_start <= speaker_end <= content_end:
        return speaker_end - content_start
    else:
        raise ValueError("Bad values for the intervals")


def get_wav_duration(wav_file: str) -> float:
    """Calculate the duration of a WAV audio file.

    Args:
        wav_file (str): Path to the WAV file.

    Returns:
        float: Duration of the audio in seconds.
    """
    with wave.open(wav_file, 'rb') as wf:
        num_frames = wf.getnframes()
        frame_rate = wf.getframerate()
        duration_seconds = num_frames / frame_rate
    return duration_seconds


def extract_turns(input_file: str, duration: int) -> Tuple[str, List[str], Dict[Tuple[float, float], str], List[Tuple[str, float, float]], List[int], List[float]]:
    """Extract speaker turns from a JSON transcript file.

    Args:
        input_file (str): Path to the input JSON file containing transcript data.
        duration (int): Duration of the associated audio file.

    Returns:
        Tuple: Contains the full transcript, list of speaker turns, 
               mapping of timestamps to content, turns times, flags, and confidence scores.
    """
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    transcript_text = data["results"]["transcripts"][0]["transcript"]
    items = data["results"]["items"]
    segments = []
    current_speaker = items[0]["speaker_label"]
    segment = []
    flags = [int(item["speaker_label"].replace("spk_", "")) for item in items]
    confidence_scores = [item["alternatives"][0]["confidence"] for item in items]
    timestamp_to_content = dict()

    for i, item in enumerate(items):
        speaker = item["speaker_label"]
        content = item["alternatives"][0]["content"]
        confidence_score = item["alternatives"][0]["confidence"]

        # Handle pronunciation and punctuation types
        if item["type"] == "pronunciation":
            start_time = float(item["start_time"])
            end_time = float(item["end_time"])
        elif item["type"] == "punctuation":
            start_time = end_time if 'end_time' in locals() else 0.0  # Ensure end_time exists
            try:
                end_time = float(items[i + 1]["start_time"])
            except IndexError:
                end_time = duration

        # Build segments
        if speaker == current_speaker:
            segment.append({
                "start_time": start_time,
                "end_time": end_time,
                "mean_time": (start_time + end_time) / 2,
                "content": content,
                "confidence_score": confidence_score
            })
            timestamp_to_content[(start_time, end_time)] = content
        else:
            segments.append((current_speaker, segment))
            current_speaker = speaker
            segment = []

    # Collect text turns and their timestamps
    text_turns = [" ".join([elem["content"] for elem in segment[1]]) for segment in segments]
    turns_times = [(segment[0], segment[1][0]["start_time"], segment[1][-1]["end_time"]) for segment in segments]
    
    return transcript_text, text_turns, timestamp_to_content, turns_times, flags, confidence_scores


def align_timestamps(input_file: str, speakers_intervals: List[Tuple[float, float, str]]) -> Tuple[str, List[str], List[float], List[float], Dict[str, List[str]]]:
    """Align timestamps between content and speakers.

    Args:
        input_file (str): Path to the input JSON file containing transcript data.
        speakers_intervals (List[Tuple[float, float, str]]): List of speaker intervals with their start and end times.

    Returns:
        Tuple: Contains the full transcript, list of content, confidence scores, 
               start and end times, and a mapping of speaker transcripts.
    """
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    transcript = data["results"]["transcripts"][0]["transcript"]
    items = data["results"]["items"]
    contents = [item["alternatives"][0]["content"] for item in items]
    confidence_scores = [item["alternatives"][0]["confidence"] for item in items]

    start_times = []
    end_times = []

    for i, item in enumerate(items):
        if item["type"] == "pronunciation":
            start_times.append(float(item["start_time"]))
            end_times.append(float(item["end_time"]))
        elif item["type"] == "punctuation":
            start_times.append(end_times[-1] if end_times else 0.0)  # Avoid index error
            try:
                end_times.append(float(items[i + 1]["start_time"]))
            except IndexError:
                end_times.append(end_times[-1] + 1)  # Fallback for duration

    speakers_counter = Counter([elem[2] for elem in speakers_intervals])
    speaker_to_transcript = {speaker: [""] * speakers_counter[speaker] for speaker in speakers_counter.keys()}

    last_speaker_index = 0
    for (content_start_time, content_end_time, content) in zip(start_times, end_times, contents):
        max_result = 0.0
        max_speaker = ""
        flag = True

        for i, (speaker_start_time, speaker_end_time, speaker) in enumerate(speakers_intervals):
            result = decide_intervals(content_start_time, content_end_time, speaker_start_time, speaker_end_time)
            if flag and result == "no overlap":
                continue
            elif flag and result == "contained":
                last_speaker_index = i
                max_speaker = speaker
                break
            else:
                flag = False
                if isinstance(result, (float, int)) and result > max_result:
                    max_result = result
                    max_speaker = speaker
                    max_speaker_index = i
                else:
                    last_speaker_index = max_speaker_index
                    break

        speaker_to_transcript[max_speaker][last_speaker_index] += f"{content} "

    # Remove empty transcripts
    for speaker in speaker_to_transcript:
        speaker_to_transcript[speaker] = [elem for elem in speaker_to_transcript[speaker] if elem]

    return (transcript, contents, confidence_scores, start_times, end_times, speaker_to_transcript)
