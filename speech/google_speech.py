import speech_recognition as sr

def audio_file_to_text(filename = "/Users/jreventos/Desktop/MAI/Semester 3/CIR/MAI-CIR/speech_recognition/audio1.wav"):

    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)
        print(text)


def speech_to_audio(lang='es-ES', chatbot_face=None):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            if chatbot_face is not None: 
                chatbot_face.status_recognizing()
            query = recognizer.recognize_google(audio,language=lang)
            print(query.lower())
            return query.lower()
        except sr.UnknownValueError:
            print("Could not understand audio")
        




import pyttsx3

def text_to_speech(text =  "Koke tiene un retraso mental, pobrecito me da un poco de pena a veces.", speed = 200, vol=1.0, voice=14):
    '''
    Transform text to speech.

    :param speed: int or float (speed of the speech)
    :param vol: int or float (volume of the speech)
    :return: audio
    '''
    engine = pyttsx3.init()

    # Change voice
    voices = engine.getProperty('voices')
    # for i in range(len(voices)):
    #     print(i, voices[i])
    engine.setProperty('voice', voices[voice].id)

    # Change rate
    rate = engine.getProperty('rate')
    engine.setProperty('rate', speed) # Increase the Speed Rate x1.25

    # Change volume
    volume = engine.getProperty('volume')
    engine.setProperty('volume', vol)

    engine.say(text)
    engine.runAndWait()
    # engine.save_to_file('Text-To-Speech you want to save as a audio file', 'audio.mp3')

    engine.stop()

#text_to_speech()


# CONVERT M4A --> WAV



# import os
# import subprocess
#
#
# import logging
# import traceback

# def m4a_to_wav(input_dir = '/Users/jreventos/Desktop/MAI/Semester 3/CIR/MAI-CIR/speech_recognition',output_dir = "/Users/jreventos/Desktop/MAI/Semester 3/CIR/MAI-CIR/speech_recognition")
#     outputdir = os.path.abspath(output_dir)
#     for root, dirs, files in os.walk(input_dir):
#         for f in files:
#             path = os.path.join(root, f)
#             base, ext = os.path.splitext(f)
#             outputpath = os.path.join(outputdir, base + ".wav")
#             if ext == '.m4a':
#                 print('converting %s to %s' % (path, outputpath))
#                 status, output = subprocess.getstatusoutput('ffmpeg -i "%s" "%s"' % (path, outputpath))
#                 if status:
#                     logging.error (output)