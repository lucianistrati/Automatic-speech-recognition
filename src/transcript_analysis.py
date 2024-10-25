import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.probability import FreqDist
from textblob import TextBlob
from transformers import pipeline
from typing import Union
from collections import Counter
from src.chat_gpt import chat_gpt
import json
# import language_tool_python
from summa import summarizer
from autocorrect import Speller
from rake_nltk import Rake
from typing import List
from statistics import mean, median, mode
from gensim import corpora, models
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from nltk.corpus import stopwords
from statistics import mean
from nltk.tokenize import word_tokenize
from textstat import (
    automated_readability_index,
    coleman_liau_index,
    dale_chall_readability_score,
    difficult_words,
    flesch_kincaid_grade,
    flesch_reading_ease,
    gulpease_index,
    gunning_fog,
    linsear_write_formula,
    osman,
    smog_index,
)
from textstat.textstat import textstatistics
from src.utils import extract_turns

nltk.download("stopwords")
nltk.download("punkt")

sentiment_pipeline = pipeline("sentiment-analysis")
import wave

import nltk
nltk.download('vader_lexicon')


def get_wav_duration(wav_file):
    with wave.open(wav_file, 'rb') as wf:
        # Get the number of frames and the frame rate
        num_frames = wf.getnframes()
        frame_rate = wf.getframerate()

        # Calculate the duration in seconds
        duration_seconds = num_frames / frame_rate

    return duration_seconds


def syllables_count(text: str) -> int:
    return int(textstatistics().syllable_count(text))


def poly_syllable_count(text: str) -> float:
    text_tokens = word_tokenize(text)
    poly_syllable_texts = [
        text for text in text_tokens if syllables_count(text) >= 3
    ]
    return len(poly_syllable_texts) / len(text_tokens)


def text_difficulty(text: str) -> float:
    # the smaller the number the more difficult the text to understand is
    scaling_factors = {
        "flesch_reading_ease": 100.0,
        "flesch_kincaid_grade": 20.0,
        "smog_index": 20.0,
        "coleman_liau_index": 20.0,
        "automated_readability_index": 20.0,
        "dale_chall_readability_score": 10.0,
        "difficult_words": 1.0,
        "linsear_write_formula": 12.0,
        "gunning_fog": 20.0,
        "gulpease_index": 100.0,
        "osman": 1.0,
        "poly_syllable_count": 1.0,
    }
    text_diff_features = [
        flesch_reading_ease(text) / scaling_factors["flesch_reading_ease"],
        flesch_kincaid_grade(text) / scaling_factors["flesch_kincaid_grade"],
        smog_index(text) / scaling_factors["smog_index"],
        coleman_liau_index(text) / scaling_factors["coleman_liau_index"],
        automated_readability_index(text)
        / scaling_factors["automated_readability_index"],
        dale_chall_readability_score(text)
        / scaling_factors["dale_chall_readability_score"],
        difficult_words(text) / scaling_factors["difficult_words"],
        linsear_write_formula(text) / scaling_factors["linsear_write_formula"],
        gunning_fog(text) / scaling_factors["gunning_fog"],
        gulpease_index(text) / scaling_factors["gulpease_index"],
        osman(text) / scaling_factors["osman"],
        poly_syllable_count(text) / scaling_factors["poly_syllable_count"],
    ]
    text_diff_features = [min(max(0.0, feature), 100.0) for feature in
                          text_diff_features]
    return float(min(max(0.0, mean(text_diff_features)), 100.0))


def predict_sentiment_nltk(text: str, discretize: bool = False) -> Union[float, -1, 0,
1]:
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)['compound']

    if discretize:
        if sentiment >= 0.66:
            sentiment = "POSITIVE"
        elif sentiment <= -0.66:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"

    return sentiment


def predict_sentiment_transformers(text: str):
    return sentiment_pipeline([text])[0]["label"]


def analyse_keywords_frequency(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    tokens = [token for token in tokens if token.isalpha() and token not in
              stopwords.words("english")]
    # Count keyword frequency
    keyword_frequency = Counter(tokens)

    return keyword_frequency


def analyze_transcripts(transcripts):
    # Tokenize and preprocess transcripts
    words = [word.lower() for transcript in transcripts for word in
             word_tokenize(transcript)]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Calculate word frequency
    fdist = FreqDist(words)

    # Plot word frequency distribution
    fdist.plot(30, cumulative=False)
    plt.title("Top 30 Most Frequent Words")
    plt.savefig("images/top_most_frequent_words.png")
    plt.show()

    # Sentiment analysis
    sentiments = [TextBlob(transcript).sentiment.polarity for transcript in transcripts]

    # Plot sentiment trends
    plt.plot(sentiments)
    plt.title("Sentiment Trends")
    plt.xlabel("Transcript Index")
    plt.ylabel("Sentiment Polarity")
    plt.savefig("images/sentiment_trends.png")
    plt.show()


def temporal_confidence_score_distribution(confidence_scores, duration, flags):
    """
    Visualize the distribution of confidence scores over time.

    Parameters:
    - confidence_scores (list): List of confidence scores corresponding to different
    time points.
    - duration (int): Total duration of the conversation in seconds.
    - flags (list): List of categorical values (0 to n) for each confidence score.
    """
    # Calculate time intervals based on the number of confidence scores and the total
    # duration
    time_intervals = [i * duration / len(confidence_scores)
                      for i in range(len(confidence_scores))]

    # Create a line plot to visualize the temporal distribution of confidence scores
    for flag_value in set(flags):
        indices = [i for i, flag in enumerate(flags) if flag == flag_value]
        marker = get_marker(flag_value)
        linestyle = get_linestyle(flag_value)
        color = get_color(flag_value)
        plt.plot([time_intervals[i] for i in indices], [confidence_scores[i] for i in indices],
                 marker=marker, linestyle=linestyle, color=color, label=f'Flag {flag_value}')

    # Set labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Confidence Score')
    plt.title('Temporal Confidence Score Distribution')

    # Display legend
    plt.legend()

    # Display the plot
    plt.savefig("images/temp_conf_score_distrib.png")
    plt.show()


def get_marker(flag_value):
    # Define a mapping from flag values to markers
    marker_mapping = {0: 'o', 1: 's', 2: '^', 3: 'v'}
    return marker_mapping.get(flag_value, 'o')


def get_linestyle(flag_value):
    # Define a mapping from flag values to linestyles
    linestyle_mapping = {0: '-', 1: '--', 2: '-.', 3: ':'}
    return linestyle_mapping.get(flag_value, '-')


def get_color(flag_value):
    # Define a mapping from flag values to colors
    color_mapping = {0: 'b', 1: 'g', 2: 'r', 3: 'c'}
    return color_mapping.get(flag_value, 'b')



def plot_sentiment_analysis(timestamp_to_sentiment_score: Dict[float, float]):
    """
    Visualize sentiment scores over time.

    Parameters:
    - timestamp_to_sentiment_score (Dict): Dictionary mapping timestamps to sentiment scores.
    """

    # Extract timestamps and sentiment scores
    timestamps, sentiment_scores = zip(*timestamp_to_sentiment_score.items())

    # Create a line plot to visualize sentiment scores over time
    plt.plot(timestamps, sentiment_scores, marker='o', linestyle='-', color='b')

    # Set labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Analysis Over Time')
    plt.savefig("images/sent_analysis_over_time.png")
    # Display the plot
    plt.show()


def plot_wordcloud(transcript: str):
    """
    Generate and display a word cloud from a transcript.

    Parameters:
    - transcript (str): Text transcript for generating the word cloud.
    """

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(transcript)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.savefig("images/word_cloud.png")
    plt.show()


def plot_transcription_length_over_time(timestamps_to_text: Dict[Tuple[float, float],
    str], duration: float):
    """
    Visualize the length of transcriptions over time.

    Parameters:
    - timestamps_to_text (Dict): Dictionary mapping timestamp segments to corresponding text.
                                Each entry has the format (start_time, end_time): "transcription_text".
    - duration (float): Total duration of the conversation in seconds.
    """

    # Extract start times, end times, and text
    start_times, end_times, transcriptions = zip(*[(start, end, text) for (start, end), text in timestamps_to_text.items()])

    # Calculate durations for each transcription
    durations = [end - start for start, end in zip(start_times, end_times)]

    # Create a line plot to visualize transcription lengths over time
    plt.plot(start_times, durations, marker='o', linestyle='-', color='b')

    # Set labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Transcription Length (seconds)')
    plt.title('Transcription Length Over Time')

    # Annotate text at each data point
    for i, txt in enumerate(transcriptions):
        plt.annotate(txt, (start_times[i], durations[i]), textcoords="offset points", xytext=(0, 5), ha='center')

    plt.savefig("images/transcription_length_over_time.png")
    # Display the plot
    plt.show()


def keyword_detection(transcript: str) -> List[str]:
    """
    Perform keyword detection in a transcript using the RAKE algorithm.

    Parameters:
    - transcript (str): Input transcript.

    Returns:
    - List[str]: List of detected keywords.
    """
    # Initialize Rake with a predefined stopword list
    rake = Rake()

    # Extract keywords from the transcript
    rake.extract_keywords_from_text(transcript)

    # Get the ranked keywords
    ranked_keywords = rake.get_ranked_phrases()

    return ranked_keywords


def topic_modelling(transcript: str) -> Dict[str, str]:
    """
    Perform topic modeling on a transcript using Latent Dirichlet Allocation (LDA).

    Parameters:
    - transcript (str): Input transcript.

    Returns:
    - Dict[str, str]: Mapping of each sentence to a topic.
    """
    # Tokenize the transcript into sentences
    sentences = [sentence.strip() for sentence in transcript.split('.') if sentence]

    # Tokenize each sentence into words
    tokenized_sentences = [sentence.split() for sentence in sentences]

    # Create a dictionary and a corpus
    dictionary = corpora.Dictionary(tokenized_sentences)
    corpus = [dictionary.doc2bow(sentence) for sentence in tokenized_sentences]

    # Apply LDA model
    lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary)

    # Map each sentence to a topic
    sentence_topic_mapping = {}
    for i, sentence in enumerate(sentences):
        bow = dictionary.doc2bow(sentence.split())
        topic_distribution = lda_model.get_document_topics(bow)
        dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
        sentence_topic_mapping[sentence] = f"Topic {dominant_topic + 1}"

    return sentence_topic_mapping


def turn_duration_statistics(turns: List[str], turns_times: List[Tuple[str, float,
float]]):
    flags = [turn[0] for turn in turns_times]
    for flag in set(flags):
        turns_durations = [len(turn) for i, turn in enumerate(turns) if turns_times[
            i][0] == flag]
        speaking_times = [turn[2] - turn[1] for turn in turns_times if turn[0] == flag]
        print("#" * 30)
        print(f"Speaker {flag} statistics")

        print("Verbal debit (number of characters):")
        print("Mean answer:", round(mean(turns_durations), 4))
        print("Median answer:", round(median(turns_durations), 4))
        print("Mode answer:", round(mode(turns_durations), 4))

        print("Speaking duration (time to speak):")
        print("Mean answer:", round(mean(speaking_times), 4))
        print("Median answer:", round(median(speaking_times), 4))
        print("Mode answer:", round(mode(speaking_times), 4))

        print("Speaking speed (characters per second):")
        print("Mean answer:", round(mean(turns_durations) / mean(speaking_times), 4))
        print("Median answer:", round(median(turns_durations) / median(speaking_times),
                                      4))
        print("Mode answer:", round(mode(turns_durations) / mode(speaking_times), 4))


def decide_intervals(content_start, content_end, speaker_start, speaker_end):
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


def vocabulary_analysis(text: str):
    """
    Vocabulary Analysis:
    Examine the vocabulary used by each speaker to identify any specialized terms or
    patterns that may reveal areas of expertise or interest.
    """
    prompt = (f"""Given what this speaker said throughout this conversation, what are the"
              "specialized terms the speaker used and what are the areas of expertise"
              " and interest possessed by the speaker? This is what the speaker said:"
              "{text}""")
    return chat_gpt(prompt)


def segment_narrative_structure(transcript: str):
    """
    Narrative Structure:
    Assess the narrative structure of the conversation, including the introduction,
    development, and resolution of topics.
    :return:
    """
    prompt = (f"Given this transcript: {transcript}, return a mapping with 3 entries "
              f"where the keys are 'introduction', 'development' and 'resolution'"
              f"and the values are pairs of timestamps between each of those 3 phases "
              f"occured in conversation?")
    return chat_gpt(prompt)


def extract_politeness_strategies(text: str):
    """
    Politeness Strategies:
    Analyze politeness strategies used by each speaker, such as
    expressions of gratitude, politeness markers, or requests.
    :return:
    """
    prompt = (f"Given the following {text} return a list of exressions of gratitude, "
              f"a list of politeness markers and a list of requests made in the "
              f"conversation")
    return chat_gpt(prompt)


def extract_speech_acts(transcript: str):
    num_speakers = 2
    data = json.load(chat_gpt(f"""Given this transcript: {transcript}. Return a json 
                              "dictionary with {num_speakers} * 3 "
                              "entries where the keys follow these 3 patterns: 
                              "speaker_i_to_questions, speaker_i_to_requests, 
                              "speaker_i_to_affirmations for each of the {num_speakers} 
                              "speakers the values are the questions, the requests                              
                              "and the affirmations respectively posed by that 
                              "speaker and written as a List[str] for each value."""))
    return data



def identify_humor(text: str):
    return chat_gpt(f"Given this text: {text} return as a list of strings all the "
                    f"instances of humor present in the text along with which "
                    f"speaker stated that affirmation if you know that")


def identify_sarcasm(text: str):
    return chat_gpt(f"Given this text: {text} return as a list of strings all the "
                    f"instances of sarcasm present in the text along with which "
                    f"speaker stated that affirmation if you know that")


def classify_communication_style(text: str):
    return chat_gpt(f"Given this text: {text} classify the communication style with "
                    f"the following options")


def classify_agreeableness(text: str) -> float:
    return chat_gpt(f"Given this text: {text} rank with a score from 0.0 to 1.0 an "
                    f"agreeableness metric "
                    f"which "
                    f"measures how agreeableness the conversation was")


def count_interruptions(text: str):
    # prompt to decide something is an interruption or not
    return chat_gpt(f"Given this text: {text} return as a list all the interruptions "
                    f"that were made during this conversation")


def extract_repetitions_and_agreements(text: str) -> List[str]:
    """
    Repetition and Agreement:
    Look for instances of repetition or agreement to identify common themes or points of
     emphasis in the conversation.
    :return:
    """
    return chat_gpt(f"Given this text: {text}, look for instances of repetition or "
                    f"agreement to identify common themes or points of emphasis in the"
                    f"conversation and return them as a list of strings")


def identify_defensive_langauge(text: str) -> List[str]:
    return chat_gpt(f"Given this text: {text} return as a list of strings all the "
                    f"instances of defensive languages")


def identify_conflicts(text: str) -> List[str]:
    return chat_gpt(f"Given this text: {text} return as a list of strings with all "
                    f"the conflicts mentioned in the text")


def identify_points_of_tension(text: str) -> List[str]:
    return chat_gpt(f"Given this text: {text} return as a list of strings all the "
                    "points of tension mentioned during this conversation between two"
                    "persons")


def predict_sentiment_llm(text: str):
    return chat_gpt(f"Given this text: {text}, predict the sentiment out of 3 "
                    f"possibile choices: NEGATIVE, NEUTRAL or POSITIVE, write just "
                    f"that word and nothing else")


def analyse_transcript(transcript_file_path: str):
    print("#" * 60)
    print(transcript_file_path)
    with open(transcript_file_path, "r") as f:
        transcript_text = f.read()

    analyze_transcripts(transcript_text)

    keyword_frequency = analyse_keywords_frequency(transcript_text)
    print("Keyword Frequency:", keyword_frequency)

    nltk_sentiment = predict_sentiment_nltk(transcript_text)
    transformers_sentiment = predict_sentiment_transformers(transcript_text)
    # llm_sentiment = predict_sentiment_llm(transformers_sentiment)

    print("NLTK Sentiment:", nltk_sentiment)
    print("Transformers sentiment:", transformers_sentiment)
    # print("LLM sentiment:", llm_sentiment)

    """
    openai.error.RateLimitError: You exceeded your current quota, please check your
    plan and billing details. For more information on this error, read the docs:
    https://platform.openai.com/docs/guides/error-codes/api-errors.
    """
    plot_wordcloud(transcript_text)
    vocabulary_analysis(transcript_text)
    segment_narrative_structure(transcript_text)
    extract_politeness_strategies(transcript_text)
    extract_speech_acts(transcript_text)
    topic_modelling(transcript_text)
    keyword_detection(transcript_text)
    # grammar_correction(transcript_text)
    speakers_intervals = [(0.0, 5.0, 0), (5.0, 6.2, 1), (6.2, 7.4, 0),
    (7.4, 14.700000000000001, 1), (14.700000000000001, 19.900000000000002, 0),
    (19.900000000000002, 28.700000000000003, 1),
    (28.700000000000003, 29.900000000000002, 0), (29.900000000000002, 36.1, 1),
    (36.1, 36.800000000000004, 0), (36.800000000000004, 41.6, 1),
    (41.6, 46.800000000000004, 0)]
    speakers_intervals = [(0.0, 2.7, 1), (2.7, 4.9, 0), (4.9, 6.2, 1),
                          (6.2, 7.300000000000001, 0),
                          (7.300000000000001, 14.700000000000001, 1),
                          (14.700000000000001, 19.900000000000002, 0),
                          (19.900000000000002, 28.8, 1), (28.8, 29.6, 0),
                          (29.6, 36.1, 1), (36.1, 36.7, 0), (36.7, 41.6, 1),
                          (41.6, 43.6, 0), (43.6, 46.6, 1),
                          (46.6, 46.800000000000004, 0)]
    # allign_timestamps('data/transcript_short_2_speakers_aws.json',
    # speakers_intervals)


def transcript_analysis():
    print("#" * 100)
    print("transcript_analysis")
    transcript_file_paths = ["data/transcript_google.txt",
                             "data/transcript_azure.txt",
                             "data/transcript_deepspeech.txt",
                             "data/transcript_google.txt"]
    transcript_file_paths = ["data/transcript_whisper_short_audio_2_speakers_333.txt"]
    for transcript_file_path in transcript_file_paths:
        analyse_transcript(transcript_file_path)


def main():
    duration = get_wav_duration("../data/short_audio_2_speakers.wav")
    (transcript_text, text_turns, timestamp_to_content, turns_times, flags,
     confidence_scores)\
        = (
        extract_turns(
            "../data/transcript_short_2_speakers_aws_segmented.json",
                         duration))
    print(transcript_text)
    print(text_turns)
    print(timestamp_to_content)
    print(turns_times)
    turn_duration_statistics(text_turns, turns_times)
    plot_transcription_length_over_time(timestamp_to_content, duration)
    timestamp_to_sentiment_score = dict()
    option = ["nltk", "transformers", "llm"][0]
    for (start_time, end_time), content in timestamp_to_content.items():
        if option == "nltk":
            sentiment = predict_sentiment_nltk(content)
        elif option == "transformers":
            sentiment = predict_sentiment_transformers(content)
        elif option == "llm":
            sentiment = predict_sentiment_llm(content)
        else:
            raise ValueError("Invalid option")
        timestamp_to_sentiment_score[(start_time + end_time) / 2] = sentiment
    plot_sentiment_analysis(timestamp_to_sentiment_score)
    temporal_confidence_score_distribution(confidence_scores, duration, flags)
    # transcript_analysis()


if __name__ == "__main__":
    main()
