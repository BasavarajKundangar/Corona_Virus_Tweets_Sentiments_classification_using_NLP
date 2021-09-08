from fastapi import FastAPI
from typing import Optional
import uvicorn
import numpy as np
import pickle
import re
import string
import pandas as pd
from sklearn.pipeline import Pipeline
from Model import Text
import nltk
import joblib 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
s = set(stopwords.words('english'))
app = FastAPI(title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the CoronaVirus Tweets ",
    version="0.1",)

from pydantic import BaseModel
class Text(BaseModel):
    text : str

model3 = pickle.load(open('model.pkl',"rb"))
bow_vec = pickle.load(open('bow.pkl',"rb"))

#Definitions
def remove_URL(headline_text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', headline_text)

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word,"", input_txt)
    return input_txt

# removing the punctuations
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " ")
    return text

# removing ASCII characters
def encoded(data):
    encoded_string = data.encode("ascii", "ignore")
    return encoded_string.decode()

# removing irrelevant characters
def reg(data):
    regex = re.compile(r'[\r\n\r\n]')
    return re.sub(regex, '', data)

#removing multi spaces
def spaces(data):
    res = re.sub(' +', ' ',data)
    return res

# removing emojis
def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

# Removing irrelevant words in clean_t column
words = set(nltk.corpus.words.words())

def clean_sent(sent):
    return " ".join(w for w in nltk.wordpunct_tokenize(sent) \
     if w.lower() in words or not w.isalpha())

# Removing Stopwords
def remove_stopwords(data):
    txt_clean=[w for w in data if w not in s]
    return txt_clean

@app.get("/greet/{text}")
def greeting(text:str):
    return {"Hi {} welcome to twitter Sentmental Analysis".format(text)}


@app.post("/predict")
def Predict_Sentiment(item:Text):
    input_text = item.text
    data_frame=pd.DataFrame([input_text],columns=['text'])
    data_frame['text'] = data_frame['text'].apply(str)
    data_frame['text'] = np.vectorize(remove_pattern)(data_frame['text'],'@[\w]*')
    data_frame['text'] = data_frame["text"].apply(remove_URL)
    data_frame['text'] = data_frame['text'].apply(remove_punctuations)
    data_frame['text'] = data_frame['text'].str.replace("[^a-zA-Z]", " ")    # removing the numeric characters
    data_frame['text'] = data_frame['text'].str.lower()                        # to convert into lower case
    data_frame['text'] = data_frame['text'].apply(reg) 
    data_frame['text'] = data_frame['text'].apply(spaces)
    data_frame['text'] = data_frame['text'].apply(remove_emojis)
    data_frame['text'] = data_frame['text'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
    data_frame['text'] = data_frame['text'].apply(clean_sent)
    data_frame['text'] = data_frame['text'].apply(lambda x: nltk.word_tokenize(x)) 
    data_frame['text'] = data_frame['text'].apply(lambda x: remove_stopwords(x))
    data_frame['text'] = data_frame['text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    data_frame['text'] = data_frame['text'].apply(str)
    bow1 = bow_vec.transform(data_frame['text'])
    final = pd.DataFrame(bow1.toarray())
    my_prediction = model3.predict(final)
    output = int(my_prediction[0])
    # output dictionary
    sentiments = {-1: "Negative \U0001F61E", 1: "Positive \U0001F603",0: "Neutral \U0001F610"}

    # show results
    result = {sentiments[output]}
    return result
