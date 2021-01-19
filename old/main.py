from transformers import  BertModel, BertTokenizer
import torch
import torch.nn as nn
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
        model_path = '/Users/jreventos/Desktop/MAI/Semester 3/CIR/MAI-CIR/BERT_sentiment.pt'
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
        self.change_status('thinking')
        self.change_mood('start_blinking')
        score = self.predict_sentiment(utterance)

        # print(score)
        self.status_custom(utterance)
        if '?' not in utterance:
            if score <= 0.55 and score > 0.45:
                self.change_mood('neutral')
            elif score <= 1 and score > 0.55:
                self.change_mood('happy')
            elif score <= 0.45 and score > 0:
                self.change_mood('sad')
        else:
            self.change_mood('confused')

    def start_blinking(self):
        self.change_mood('start_blinking')

    def stop_blinking(self):
        self.change_mood('stopt_blinking')

    def status_listening(self):
        self.change_status('listening')

    def status_thinking(self):
        self.change_status('thinking')

    def status_none(self):
        self.change_status('')

    def neutral(self):
        self.change_mood('neutral')

    def happy(self):
        self.change_mood('happy')

    def sad(self):
        self.change_mood('sad')

    def angry(self):
        self.change_mood('angry')

    def focused(self):
        self.change_mood('focused')

    def confused(self):
        self.change_mood('confused')






chatbot = ChatbotFace()
chatbot.mood_prediction('This film is great')

# SPEECH
from speech import google_speech as speech
# Test
#speech.speech_to_audio(lang="en-US")

# TRANSLATION
from google_trans_new import google_translator
translator = google_translator()
translate_text = translator.translate('สวัสดีจีน', lang_tgt='es')
print(translate_text)


# CHATBOT
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from google_trans_new import google_translator


class Chatbot:
    def __init__(self, lang='en'):
        print('Loading model and tokenizers...')
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.face = ChatbotFace()

        self.reset_chatbot(lang=lang)
        print('Chatbot ready.')

    def reset_chatbot(self, lang='en'):
        self.chat_history_ids = self.tokenizer.encode(self.tokenizer.bos_token, return_tensors='pt')
        if lang == 'es':
            self.voice = 14  # 0: spanish female, 1: english female, 2: english male
        else:
            self.voice = 0  # 0: spanish female, 1: english female, 2: english male
        self.translator = google_translator()
        self.lang = lang

    def listen_and_get_question(self):
        self.face.status_listening()
        lang = 'en-US' if self.lang == 'en' else 'es-ES'
        question = speech.speech_to_audio()
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
        self.face.happy()
        speech.text_to_speech(text, speed = 200, vol=1.0, voice=self.voice)
        self.face.neutral()

    def translate(self, text, lang):
        translated_text = self.translator.translate(text, lang_tgt=lang)
        if type(translated_text) == list:
            return translated_text[0]
        return translated_text

    def no_understand(self):
        text = "Sorry, I couldn't understand you"
        if self.lang == 'es':
            text = self.translete(text, lang='es')
        self.speak(text)

    def run_chat(self):
        self.reset_chatbot(lang=self.lang)
        self.face.neutral()
        question = ''
        while (question != 'goodbye'):
            question = self.listen_and_get_question()
            if question is None:
                print('Fallo')
                self.no_understand()
                continue
            print('Question:', question)
            answer = self.generate_answer(question)
            print('Answer:', answer)
            self.speak(answer)

            #             self.reset_chatbot(lang=self.lang)
            print(self.chat_history_ids)

chatbot = Chatbot(lang='en')
chatbot.run_chat()