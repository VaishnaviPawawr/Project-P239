from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import time
import re
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

stopwords_set = set(stopwords.words('english'))
stopwords_set1=list(stopwords_set)
text=""
text1=""
filename="SVCmodel1.pkl"
filename1="word2vec_model1.pkl"

def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = tokens
    return filtered_tokens

with open(filename1, 'rb') as file:
    word2vec_model = pickle.load(file)

def display_sarcastic_remark(remark):
    st.title(remark)
    time.sleep(0.1)

st.header('Sentiment Analysis')
with st.title('Analyze Text'):
	text = st.text_input('Text here: ')
if text:
	text1=text
	blob = TextBlob(text)
#st.write('Polarity: ', round(blob.sentiment.polarity,2))
#st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))
plt.figure(figsize = (20,20))

redunant_df = pd.read_csv("reduced_words.csv")
redunant=set(redunant_df["words"])

def cleaning_reduntant(text):
    return " ".join([word for word in str(text).split() if word not in redunant])

text1=cleaning_reduntant(text1)

def annotate_words(sentence, word_set):
    annotated_sentence = sentence

    for word in word_set:
        annotated_sentence = annotated_sentence.replace(word, f'<span style="background-color: yellow;">{word}</span>')

    # Display the annotated sentence
    st.markdown(annotated_sentence, unsafe_allow_html=True)

target_sentence = "I like to eat apples and bananas."
custom_words = ["apples", "bananas"]

annotate_words(text, stopwords_set1)

def word2vec_features(text, model, num_features):
    feature_vector = np.zeros((num_features,), dtype="float32")
    num_words = 0
    for word in text:
        if word in model.wv.key_to_index:
            num_words += 1
            feature_vector = np.add(feature_vector, model.wv[word])
    if num_words > 0:
        feature_vector = np.divide(feature_vector, num_words)
    return feature_vector.reshape(-1, num_features)



if(text1!=""):
    st.title("Cleaned Text")
    text1 = re.sub('((www.[^s]+)|(https?://[^s]+))|(http?://[^s]+)', '',text1)
    tknzr = TweetTokenizer(strip_handles=True)
    text1=tknzr.tokenize(text1)
    text1=str(text1)
    text1=re.sub(r'[^a-zA-Z0-9\s]', '', text1)
    text1=cleantext.clean(text1, clean_all= False, extra_spaces=True ,stopwords=True ,lowercase=True ,numbers=True , punct=True)
    st.write(text1)

filtered_tokens=preprocess_text(text1)

#embed_text=word2vec_features(filtered_tokens, word2vec_model, num_features=100)
#X_test_2d = np.stack(embed_text)
#X_test_embed = scaler.fit_transform(X_test_2d)

with open(filename, 'rb') as file:
    model = pickle.load(file)
#unseen_tweets=[text1]
#unseen_df=pd.DataFrame(unseen_tweets)
#unseen_df.columns=["Unseen"]

#X_test = vectorizer.transform(unseen_tweets)
#y_pred = model.predict(X_test_2d)
y_pred=0
if text!="":
    if(y_pred==0):
        remark = "That's Figurative!😄"
        display_sarcastic_remark(remark)
    if(y_pred==1):
        remark = "That's Irony!😏"
        display_sarcastic_remark(remark)
    if(y_pred==2):
        remark = "That's Regular!😐"
        display_sarcastic_remark(remark)
    if(y_pred==3):
        remark = "That's Sarcasm!🙃"
        display_sarcastic_remark(remark)
else:
    st.write(text1)
    remark = "No Words to Analyze"
    display_sarcastic_remark(remark)