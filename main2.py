from transformers import  BertModel, BertTokenizer
import requests
from sentiments.BERTGRU_model import BERTGRUSentiment


class ChatbotFace:
    '''Chatbot face according to sentiment analysis. '''
    chatbot_mood_API = 'https://chatbot-mood.herokuapp.com/mood'
    chatbot_status_API = 'https://chatbot-mood.herokuapp.com/internal_state'

    def change_mood(self, mood):
        req_obj = {'action': mood}
        x = requests.post(self.chatbot_mood_API, data=req_obj)
        # print(x)

    def change_status(self, status):
        req_obj = {'status': status}
        x = requests.post(self.chatbot_status_API, data=req_obj)

    def status_custom(self, text):
        self.change_status(text)

    def predict_sentiment(self, sentence):
        model_path = 'BERT_sentiment.pt'
        max_input_length =  512
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased')
        model = BERTGRUSentiment(bert,
                                 hidden_dim=256,
                                 output_dim=1,
                                 n_layers=2,
                                 bidirectional=True,
                                 dropout=0.25)
        model.to(device)
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        model.eval()
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:max_input_length - 2]

        # Special tokens:
        init_token = tokenizer.cls_token  # Initiate sentence
        eos_token = tokenizer.sep_token  # End of sentence

        # Special tokens idx
        init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
        eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)

        # Sentence index
        indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]

        # To tensor
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)
        prediction = torch.sigmoid(model(tensor))

        return prediction.item()

    def mood_prediction(self, utterance):

        score = self.predict_sentiment(utterance)

        self.status_custom(utterance)

        if 'how' or 'what' or 'why' or 'which' or 'when' in utterance:
            self.change_mood('confused')
        else:
            if score <= 0.55 and score > 0.45:
                self.change_mood('neutral')
            elif score <= 1 and score > 0.55:
                self.change_mood('happy')
            elif score <= 0.45 and score > 0:
                self.change_mood('sad')


    def start_blinking(self):
        self.change_mood('start_blinking')

    def stop_blinking(self):
        self.change_mood('stop_blinking')

    def status_listening(self):
        self.change_status('listening')

    def status_thinking(self):
        self.change_status('thinking')

    def status_none(self):
        self.change_status('')






# SPEECH
from speech import google_speech as speech
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead
from google_trans_new import google_translator
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import webbrowser as web

class Chatbot:
    def __init__(self, lang='en'):


        web.open("https://gonzalorecio.com/chatbot/robot.html")

        print('Loading model and tokenizers...')

        # Initialize chatbot interface
        self.face = ChatbotFace()

        # Define chatbot voice
        self.define_voice(lang=lang)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        #self.tokenizer = AutoTokenizer.from_pretrained("ncoop57/DiGPTame-medium")
        #self.model = AutoModelWithLMHead.from_pretrained("ncoop57/DiGPTame-medium")

        # Display instructions:
        #self.initial_instructions(lang=lang)

        # Reset chatbot
        self.reset_chatbot(lang=lang)
        #self.speak('Welcome to our chatbot interface. In a few seconds you can start a conversation with me. Now I am loading the system, I will be ready in a second!')

        #Welcome message
        # if lang == 'en':
        #     self.face.status_custom('Hey I am chatbot!')
        #     self.speak('Hey! I am a chatbot developed by Gonzalo Recio and Jana Reventós.   '
        #                'In a few seconds you can start a conversation with me.   '
        #                'In the blink of an eye, I will be ready.')
        # else:
        #     self.face.status_custom('Hola, soy un chatbot!')
        #     self.speak(
        #         'Hola! Soy un chatbot creado por Gonzalo Recio y Jana Reventós.   '
        #         'En unos segundos podrás empezar una conversación conmigo.   '
        #        'Estaré preparado en menos de un abrir y cerrar de ojos!')

        print('Chatbot ready.')


    def initial_instructions(self,lang='en'):

        if lang == 'en':
            text1 = 'Please read the following instructions (1-8) before starting the conversation:'
        else:
            text1 = 'Porfavor lea las siguientes instrucciones (1-8) antes de empezar la conversación:'
        self.face.status_custom(text1)
        self.speak(text1)

        # Instruction 1
        if lang == 'en':
            inst1  = '1. This is a test for our CIR subject final project.'
        else:
            inst1 = '1. Esto es un test para nuestro proyecto de la asignatura de CIR.'
        self.face.status_custom(inst1)
        self.speak(inst1)

        # Instruction 2
        if lang == 'en':
            inst2 = '2. We aim to test this conversational system with real users.'
        else:
            inst2 = '2. Nuestro objetivo es realizar una prueba de nuestro sistema de conversación con usuarios reales.'

        self.face.status_custom(inst2)
        self.speak(inst2)

        # Instruction 3
        if lang == 'en':
            inst3 = '3. This session will be recorded to test the reaction of the users during the interaction with the chatbot.'
        else:
            inst3 = '3. Esta sesión será grabada para evaluar la reacción de los usuarios durante la interacción con el chatbot.'

        self.face.status_custom(inst3)
        self.speak(inst3)

        # Instruction 4
        if lang == 'en':
            inst4 = '4. When you read the phrase: I AM LISTENING in the screen, you have to speak.'
        else:
            inst4 = '4. Cuando leas la frase: I AM LISTENING en la pantalla, deberás hablar.'
        self.face.status_custom(inst4)
        self.speak(inst4)

        # Instruction 5
        if lang == 'en':
            inst5 = '5. Please wait while the chatbot is thinking an answer. You will read the word THINKING in the screen.'
        else:
            inst5 = '5. Porfavor, espera a que el chatbot pinse su respuesta. Vas a leer la palabra THINKING en la pantalla.'
        self.face.status_custom(inst5)
        self.speak(inst5)

        # Instruction 6
        if lang == 'en':
            inst6 = '6. Once the answer is ready, the chatbot will talk to you.'
        else:
            inst6 = '6. Una vez la respuesta este preparada, el chatbot te va a contestar.'
        self.face.status_custom(inst6)
        self.speak(inst6)

        # Instruction 7
        if lang == 'en':
            inst7 = '7. Then, you can answer the chatbot again.'
        else:
            inst7 = '7. Cuando hayas escuchado la respuesta podrás volver a responder al chatbot.'
        self.face.status_custom(inst7)
        self.speak(inst7)

        # Instruction 8
        if lang == 'en':
            inst8 = '8. At the end, a screen with a questionnaire to fill out will be displayed.'
        else:
            inst8 = '8. Al final de la conversación, un cuestionario a rellenar aparecerá. .'
        self.face.status_custom(inst8)
        self.speak(inst8)

    def define_voice(self,lang='en'):
        if lang == 'es':
            self.voice = 14  # 0: spanish female, 1: english female, 2: english male
        else:
            self.voice = 0  # 0: spanish female, 1: english female, 2: english male

    def reset_chatbot(self, lang='en'):
        self.chat_history_ids = self.tokenizer.encode(self.tokenizer.bos_token, return_tensors='pt')
        self.translator = google_translator()
        self.lang = lang

    def listen_and_get_question(self):
        self.face.status_listening()
        lang = 'en-US' if self.lang == 'en' else 'es-ES'
        question = speech.speech_to_audio()
        self.question_original = question
        if self.lang == 'es':
            question = self.translate(question, lang='en')
        self.face.status_none()
        return question

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate_answer(self, question):
        self.face.status_thinking()
        question_ids = self.tokenizer.encode(question + self.tokenizer.eos_token, return_tensors='pt')
        input_ids = torch.cat([self.chat_history_ids, question_ids], dim=-1)
        self.chat_history_ids = self.model.generate(input_ids, max_length=1000,
                                                    pad_token_id=self.tokenizer.eos_token_id)
        answer_ids = self.chat_history_ids[:, input_ids.shape[-1]:][0]
        text = self.decode(answer_ids)
        if self.lang == 'es':
            text = self.translate(text, lang='es')
        self.face.status_none()
        return text

    def get_sentiment_analysis(self, text):
        return True

    def speak(self, text):
        self.face.change_mood('happy')
        speech.text_to_speech(text, speed = 200, vol=1.0, voice=self.voice)
        self.face.change_mood('neutral')

    def translate(self, text, lang):
        translated_text = self.translator.translate(text, lang_tgt=lang)
        if type(translated_text) == list:
            return translated_text[0]
        return translated_text

    def no_understand(self):
        text = "Sorry, I couldn't understand you"
        if self.lang == 'es':
            text = self.translate(text, lang='es')
        self.speak(text)

    def run_chat(self):

        self.face.change_mood('neutral')
        # Chatbot welcome
        self.reset_chatbot(lang=self.lang)
        question = ''
        previous_answer =''
        while (question != 'goodbye'):
            question = self.listen_and_get_question()
            if question is None:
                print('Fallo')
                self.no_understand()
                continue
            print('User:', self.question_original)
            answer = self.generate_answer(question)
            self.face.mood_prediction(answer)
            self.face.status_custom(answer)
            print('CIRREL chatbot:', answer)
            self.speak(answer)
            if answer != previous_answer:
                previous_answer = answer
                self.chat_history_ids = self.chat_history_ids[:, -50:]
                continue
            else:

                if self.lang == 'en':
                    self.face.status_custom('This conversation has ended')
                    self.speak('This conversation has ended. It has been a pleasure talking with you. Thank you! ')
                    print('Conversation ended')
                else:
                    self.face.status_custom('Esta conversación ha terminado.')
                    self.speak('Esta conversación ha terminado. Ha sido un placer hablar contigo. Muchas gracias!')
                    print('Conversación terminada.')

                break
        if self.lang == 'en':
            web.open('https://docs.google.com/forms/d/e/1FAIpQLSfZ4U1pSPwSFOsC3bl46QtRW3HoH1-6XQbUOMH1u0Wjx4lnfg/viewform?usp=sf_link')
        else:
            web.open("https://docs.google.com/forms/d/e/1FAIpQLSdg68q2xtkrdaabnCpIjIQylo7h3Opywpkjy-OqXSCSeq0_cg/viewform?usp=sf_link")
        self.face.status_none()
            # self.reset_chatbot(lang=self.lang)
            # print(self.chat_history_ids)


chatbot = Chatbot(lang='es')
chatbot.run_chat()