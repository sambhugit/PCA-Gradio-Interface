import gradio as gr
import json 
import numpy as np
import string
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import random
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import matplotlib
from matplotlib import pyplot as plt
import Google_drive_downloader
import os
############################################### Loaders and Variables ##########################################################

sentiment = np.array([0])
res = np.array([0])
emotion_var=[0]*9
Covid_Status_Fixed = "default"   
inp = []

# Path Directory

current_dir = os.path.dirname(os.path.realpath('Gradio_Func_File.py'))

# load chatbot dataset
with open(os.path.join(current_dir,'intents.json')) as file:
    data = json.load(file)

# load chatbot model

model_chat = keras.models.load_model(os.path.join(current_dir,'best_model_ES_CB_new2.hdf5'))

# load chatbot tokenizer object
with open(os.path.join(current_dir,'tokenizer_new2.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)

# load label encoder object
with open(os.path.join(current_dir,'label_encoder_new2.pickle'), 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# parameters
max_len = 20 
max_review_length = 200

# load sentiment analysis model
weight_path = os.path.join(current_dir,'sentiment_weight_file.hdf5')
model_sa = load_model(weight_path)

# load sentiment analysis tokenizer object
with open(os.path.join(current_dir,'tokenizer_sa_new1.pickle'), 'rb') as handle:
    tokenizer_sa = pickle.load(handle)

############################################ Secondary Sentence Cleaner Functions #################################################

def full_remove(x, removal_list):

    for w in removal_list:
        x = x.replace(w, ' ')
    return x

def removeStopWords(stopWords, txt):
    newtxt = ' '.join([word for word in txt.split() if word not in stopWords])
    return newtxt

def stem_with_porter(words):
    porter = nltk.PorterStemmer()
    new_words = [porter.stem(w) for w in words]
    return new_words

# Main sentence cleaner function

def remove(sentences):
    ## Remove digits ##
    digits = [str(x) for x in range(10)]
    remove_digits = [full_remove(x, digits) for x in sentences]
    ## Remove punctuation ##
    remove_punc = [full_remove(x, list(string.punctuation)) for x in remove_digits]
    ## Make everything lower-case and remove any white space ##
    sents_lower = [x.lower() for x in remove_punc]
    sents_lower = [x.strip() for x in sents_lower]
    ## Remove stop words ##
    stops = ['the', 'a', 'an', 'i', 'he', 'she', 'they', 'to', 'of']
    sents_processed = [removeStopWords(stops,x) for x in sents_lower]
    porter = [stem_with_porter(x.split()) for x in sents_processed]
    sents_processed = [" ".join(i) for i in porter]

    return sents_processed
################################## Primary app.py functions #####################################
    
    
def sentiment_call(inp):
    
    result = [0]
    outcome_labels = [0,1]

    for m in inp:
        s = [m]
        seq = tokenizer_sa.texts_to_sequences(s)
        padded = pad_sequences(seq, maxlen=max_review_length) 
        pred = model_sa.predict(padded)
        result = np.append(result,outcome_labels[np.argmax(pred)])
    return result

def identify_sentiment(sentiment,emotion_var):
    
    index = [0,1]
    sentiment = np.delete(sentiment,index)
    encrypted_value_func = sentiment
    value = sum(sentiment)/(len(sentiment))
    if value >= 0.5:
        user_sentiment = "User is Not Depressed"
    else:
      user_sentiment = "User is Depressed"
    
    return user_sentiment, encrypted_value_func

# Main app.py function

def greet(User_Text, Covid_Status):
    
    inp_save = User_Text
    temp = remove([inp_save])
    plt.close()
    encrypted_value = 0

    for x in temp:
        inp_save = x
    
    global inp

    if len(inp) == 1:
        global Covid_Status_Fixed
        Covid_Status_Fixed = Covid_Status
    
    no_of_letters = 0
          
    for letter in inp_save:
        no_of_letters = no_of_letters + 1
    
    # THE BUG FIXER IF CONDITIONS FOR CHATBOT :')

    if no_of_letters > 70:
        
        plt.pie(1, colors = 'w')
        
        return  "Your text is very long for me to analyse. Please use small sentences, simple words and simple english.", plt, str(0)
                                                           
                                                                    # Case 1 Return Function. Used if user input is large.
    if no_of_letters == 0:
        
        plt.pie(1, colors = 'w')
        
        return "You must type something it the box", plt , str(0)
                                                                    # Case 2 Return Function. Used if user haven't given any input.
    global res, sentiment, emotion_var
    
    if inp_save.lower() not in ("end","end.","end,","end:","end","end","end","end ","end?","end;","end..",
                                 "end!","end*","end~","end  ","end","end    ",   "end"):
        
        inp.append(inp_save)
        result = model_chat.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp_save]),
                                              truncating='post', maxlen=max_len))                                  
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        
        for i in data['intents']:
                if i['tag'] == tag:
                    reply = np.random.choice(i['responses'])
                    plt.pie(1, colors = 'w')
                    encrypted_default_value = 0
                    
                    if tag == 'boredom':
                        emotion_var[0] = emotion_var[0] + 1
                    elif tag == 'calmness': 
                        emotion_var[1] = emotion_var[1] + 1
                    elif tag == 'anxiety': 
                        emotion_var[2] = emotion_var[2] + 1
                    elif tag == 'terror':
                        emotion_var[3] = emotion_var[3] + 1
                    elif tag == 'absurd':
                        emotion_var[4] = emotion_var[4] + 1
                    elif tag == 'care':
                        emotion_var[5] = emotion_var[5] + 1
                    elif tag == 'sad':
                        emotion_var[6] = emotion_var[6] + 1
                    elif tag == 'angry':
                        emotion_var[7] = emotion_var[7] + 1
                    elif tag == 'happy':
                        emotion_var[8] = emotion_var[8] + 1
                    return reply, plt, str(encrypted_default_value) # Case 3 Return Function. It is used for personal chat with bot
    else:
      res = np.array([0])

      # THE BUG FIXER IF CONDITIONS FOR SENTIMENT ANALYSIS :')

      if len(inp) == 1:
          
          inp = []
          sentiment = np.array([0])
          user_sentiment = ''
          encrypted_value = 0
          text = ''
          emotion_var=[0]*9
          return "You haven't send enough texts", plt , str(0) # Case 4 Return Function. It is used if user inputted only one chat. This
                                                              # can give a div by 0 error during Sentiment Analysis.
      if emotion_var == [0]*9:
          
          inp = []
          sentiment = np.array([0])
          user_sentiment = ''
          encrypted_value = 0
          text = ''
          emotion_var=[0]*9 
          
          return "You haven't talked about your emotions or feelings or problems. Start chatting from the beginning again.", plt , str(0) # Case 5 Return Function.
                                                                                                                                          # Used if user haven't talked about his emotions yet.

      res = sentiment_call(inp)
      inp = []
      sentiment = np.array([0])
      sentiment = np.append(sentiment,res)
      user_sentiment = ''
      encrypted_value = 0
      user_sentiment, encrypted_value = identify_sentiment(sentiment,emotion_var)
      text = ''
      text = user_sentiment + ' & ' + Covid_Status_Fixed
      sum_emotion_var = sum(emotion_var,0)
    
      mylabels = ["bored "+str((emotion_var[0]/sum_emotion_var)*100),"calm "+str((emotion_var[1]/sum_emotion_var)*100),
                  "anxious "+str((emotion_var[2]/sum_emotion_var)*100),"afraid "+str((emotion_var[3]/sum_emotion_var)*100),
                  "absurd "+str((emotion_var[4]/sum_emotion_var)*100),"care "+str((emotion_var[5]/sum_emotion_var)*100),
                  "sad "+str((emotion_var[6]/sum_emotion_var)*100),"angry "+str((emotion_var[7]/sum_emotion_var)*100),
                  "happy "+str((emotion_var[8]/sum_emotion_var)*100)]
      
      mycolors = ['r','g','b','c','m','y','#FF7F50','#B8860B','#4CAF50']
      
      plt.pie(emotion_var,labels = mylabels, colors = mycolors)
      plt.legend()
      reply = text
      encrypted_value = '['+str(encrypted_value)+','+str(emotion_var)+']'
      emotion_var=[0]*9

    return reply, plt, str(encrypted_value) # Case 6 Return Function. Used to give Sentiment Analysis result to user.

# Gradio Interface Loaders
iface = gr.Interface(fn=greet, 
                     inputs=[gr.inputs.Textbox(lines=1, placeholder="Type your text here..."),
                             gr.inputs.Radio(["Currently having Covid","Infected from Covid before","Never had Covid"])], 
                     outputs=[gr.outputs.Textbox(type="str", label="ISHA"),
                              gr.outputs.Image(type="plot", plot = True, label = "Emotion Chart"),
                              gr.outputs.Textbox(type="str", label= "Encrypted Value")], 
                     title="ISHA- Your AI Bot",description="Talk freely with ISHA about your problems and emotions here. Use SUBMIT button to send your text to ISHA. Update Covid Status before you send your first text. Type keyword 'end' under USER TEXT and click SUBMIT to end the chat and show results. Please click 'FLAG' button after getting results.",
                     allow_flagging = True,
                     allow_screenshot = False,
                     flagging_dir= os.path.join(current_dir,'flagged'))
iface.launch()
