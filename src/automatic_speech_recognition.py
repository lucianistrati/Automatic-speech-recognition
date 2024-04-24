import io
import os
import wave
import json
import librosa
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr
import azure.cognitiveservices.speech as speechsdk
from google.cloud import speech_v1p1beta1 as speech
from pocketsphinx import LiveSpeech
import whisper
import boto3

def transcribe_mozilla(audio_file_path):
    # Abandoned due to installation issues with deepspeech
    pass

def transcribe_azure(audio_file_path, output_file_path, subscription_key, service_region):
    if os.path.exists(output_file_path):
        return
    
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=service_region)
    audio_input = speechsdk.audio.AudioConfig(filename=audio_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
    result = speech_recognizer.recognize_once()
    
    text = result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else "Error: {}".format(result.reason)
    
    with open(output_file_path, "w") as f:
        f.write(text)

def transcribe_google(audio_path, output_transcript_path):
    if os.path.exists(output_transcript_path):
        return
    
    client = speech.SpeechClient()

    with io.open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    with open(output_transcript_path, 'w') as transcript_file:
        for result in response.results:
            transcript = result.alternatives[0].transcript
            transcript_file.write(transcript + '\n')

def transcribe_pocketsphinx(input_file: str, output_file: str):
    if os.path.exists(output_file):
        return
    
    speech = LiveSpeech(
        audio_file=input_file,
        verbose=False,
        sampling_rate=16000,
        buffer_size=2048
    )

    transcriptions = []

    for phrase in speech:
        transcriptions.append(str(phrase))

    with open(output_file, "w") as f:
        f.write(' '.join(transcriptions))

def transcribe_whisper(input_file: str, output_file: str):
    if os.path.exists(output_file):
        return
    
    model = whisper.load_model("base")
    result = model.transcribe(input_file)
    text = result["text"]
    with open(output_file, "w") as f:
        f.write(text)

def transcribe_aws(audio_file_path, output_file, job_name, output_bucket):
    if os.path.exists(output_file):
        return
    
    transcribe = boto3.client('transcribe')

    response = transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode='en-US', 
        MediaFormat='wav',
        Media={
            'MediaFileUri': f's3://{output_bucket}/{audio_file_path}'
        },
        OutputBucketName=output_bucket
    )

    with open(output_file, 'w') as json_file:
        json.dump(response, json_file, indent=4)

def audio_spectrogram(audio_filepath):
    y, sr = librosa.load(audio_filepath)
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig("images/audiogram_spectrogram.png")
    plt.show()

def automatic_speech_recognition(input_file_path: str = 'data/audio.wav'):
    print("#" * 109)
    print("automatic_speech_recognition")

    audio_spectrogram(input_file_path)
    mozilla_transcription = transcribe_mozilla(input_file_path)
    print("Mozilla Transcription:", mozilla_transcription)

    output_transcript_path = '../data/transcript_whisper_short_audio_2_speakers_333.txt'
    transcribe_whisper(input_file_path, output_transcript_path)

    output_transcript_path = 'data/transcript_aws.txt'
    job_name = 'LongAudio'
    output_bucket = 'challengevespioai'
    region = "eu-west-3"

    # transcribe_aws(input_file_path, output_transcript_path, job_name, output_bucket)
    # print(f"Transcription Job Status: {response['TranscriptionJob']['TranscriptionJobStatus']}")

    output_transcript_path = 'data/transcript_google.txt'
    transcribe_google(input_file_path, output_transcript_path)

    output_transcript_path = 'data/transcript_pocketsphinx.txt'
    transcribe_pocketsphinx(input_file_path, output_transcript_path)

    azure_subscription_key = 'your_azure_subscription_key'
    azure_service_region = 'eastus'
    output_transcript_path = "data/transcript_azure.txt"
    transcribe_azure(input_file_path, output_transcript_path, azure_subscription_key, azure_service_region)

if __name__ == "__main__":
    automatic_speech_recognition()
