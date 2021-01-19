# MAI-CIR

## Deep Reinforcement Learning Open-domain chatbot
Master in Artificial Intelligence - UPC Cognitive Interaction with Robots Final Project



**Authors: Gonzalo Recio and Jana Reventós**

In this repository we present an  open-domain  conversational  chatbot  that  is  able  to  interact  with  users by voice through a graphical interface.  Many conversational chatbots lack of engagement, are short-sighted and tend to generate incoherent or repetitive responses making the interaction last for a short period.  To solve this, we propose a deep reinforcement learning approach to train its core language model  (GPT  Transformer)  in  order  to  improve  the  long-term  performance. 

### Main requirements
Models:
```
pip install torch
pip install transformers     (for DialoGPT model)
```
Interface (speech recognition and synthesis, translation):  
```
pip install SpeechRecognition
pip install google-speech
pip install google-trans-new
```

### Code

To run the chatbot run `python main.py` on a terminal. This command automatically opens de chatbot interface. Web-based interface of the chatbot interface can be found at: https://gonzalorecio.com/chatbot/robot.html

The code to train the model with Deep Reinforcement Learning is located at `seq2seq/seq2seq-RL.ipynb`.

### Model wights

Model weights and checkpoints from RL traning and BERT-GRU for sentiment analysis are located at https://drive.google.com/drive/folders/17BZD6kc2ATJf26cEjSYMLoXdEHp6CTdc?usp=sharing.


### Results

In the directiory `results` there are the quantitative results of the interactions and the output dialogues. Inside folder `questionnaire` we can find the results of the participant questionnaires.
