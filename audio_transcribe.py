import speech_recognition as sr
from pydub import AudioSegment

def speech_to_text(path):
    # Initializing the recognizer
    r = sr.Recognizer()

    # Checking the audio file format and converting to .wav file if it is not.
    try:
        with sr.WavFile(path) as source:              
            # extract audio data from the file
            audio = r.record(source)
    except:
        song = AudioSegment.from_file(path)
        wav_path = path.split('.')[0] + '.wav'
        song.export(wav_path, format="wav")

        with sr.WavFile(wav_path) as source:
            # extract audio data from the file
            audio = r.record(source)
    try:
        # using Google Speech Recognition for transcribing text
        # print("Transcription: " + r.recognize_google(audio))
        return r.recognize_google(audio)

    except LookupError:
        print("Could not understand audio")
        return None

