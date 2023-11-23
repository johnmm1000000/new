from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
#model=tf.keras.models.load_model("C://Users//Public//model2.h5")
from fastapi import FastAPI
from pydantic import BaseModel 
import uvicorn

#import json
#import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
model=tf.keras.models.load_model("xlnet_english_deepfakedetection.h5")
model1=tf.keras.models.load_model("arabicbertfordeepfaketextdetection.h5")
app=FastAPI()



   


class UserInput(BaseModel):
    class Config:
        arbitrary_types_allowed = True 


 


    
    text:str

    

    
    
@app.get('/')
async def index():
    return {"hola"}
import re    
def is_arabic(text):
    arabic_regex = re.compile(r'[\u0600-\u06FF]+')
    return arabic_regex.search(text) != None

def is_english(text):
    english_regex = re.compile(r'[a-zA-Z]+')
    return english_regex.search(text) != None
def tokenize_data(tokenizer, texts):
  """Tokenizes a list of text strings using the given BERT tokenizer.

  Args:
    tokenizer: A BERT tokenizer.
    texts: A list of text strings.

  Returns:
    A list of tokenized sentences, where each sentence is a list of token IDs.
  """

  tokenized_texts = []
  for text in texts:
    # Fix the error by removing the `return_tensors='pt'` argument.
    tokens = tokenizer(text, padding=True, truncation=True, max_length=128)['input_ids']
    tokenized_texts.append(tokens)

  return tokenized_texts

import transformers
from transformers import  XLNetTokenizer,AutoTokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
arabic_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
@app.post('/predict')

async def predict(UserInput:UserInput):
  
    if is_arabic(UserInput.text):

        x2 = tokenize_data(arabic_tokenizer, [UserInput.text])

# Make a prediction
        prediction_input = np.array(x2).reshape(-1)
        prediction_input=pad_sequences([prediction_input],128)


        output = model1.predict(prediction_input)




        if output > 0.5:
            predicted_class = 'human written'
        else:
            predicted_class = 'machine generated'
    elif is_english(UserInput.text):
        x2 = tokenize_data(tokenizer, [UserInput.text])
        prediction_input = np.array(x2).reshape(-1)
        prediction_input=pad_sequences([prediction_input],128)


        output = model.predict(prediction_input)




        if output > 0.5:
            predicted_class = 'human written'
        else:
            predicted_class = 'machine generated'

    
      
        
            
    

    return {'predicted_class': predicted_class, 'output': output.tolist()}