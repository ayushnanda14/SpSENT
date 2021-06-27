import os, sys
from sentiment_predict import load_model, predict_sentiment
from audio_transcribe import speech_to_text
from download_model import download_file_from_google_drive, find


model = load_model()

if(len(sys.argv) == 2):
    path = sys.argv[1]

    try:
        transcription = speech_to_text(path)
    
    except FileNotFoundError:
        print('File Not Found: please ensure that the audio file is present in the audio directory. And enter the file extension too.')
        sys.exit(0)

    sentiment = predict_sentiment(transcription, model)
    print('Transcription: ', transcription)
    print('Predicted Sentiment: ', sentiment)

while(1):
    choice = 'y'
    if choice == 'y':
        new_path = input('Enter the path to the audio file - ')
        try:
            transcription = speech_to_text(new_path)
        except FileNotFoundError:
            print('File Not Found: please ensure that the audio file is present in the audio directory. And enter the file extension too.')
            sys.exit(0)

        sentiment = predict_sentiment(transcription, model)
        print('Transcription: ', transcription)
        print('Predicted Sentiment: ', sentiment)

    choice = 'n'
    choice = input('\n\nDo you want to try another file? [y/n] (Default - n): ')
    if choice !='y':
        break
