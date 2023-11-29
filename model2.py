from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
#model=tf.keras.models.load_model("C://Users//Public//model2.h5")
from fastapi import FastAPI
from pydantic import BaseModel 
import uvicorn
import uvicorn
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile 
from keras.preprocessing import image
from io import BytesIO
import os
#import json
#import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
model=tf.keras.models.load_model("xlnet_english_deepfakedetection.h5")
model1=tf.keras.models.load_model("arabicbertfordeepfaketextdetection.h5")
app=FastAPI()
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
import PyPDF2
import docx
import transformers
from transformers import  XLNetTokenizer,AutoTokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
arabic_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("txt", "pdf", "docx")
    if not extension:
        return "File must be pdf or docx or txt format!"

    doc = await file.read()

    if doc.startswith(b'%PDF'):
        # PDF file
   

        pdf_reader = PyPDF2.PdfReader(BytesIO(doc))
        pdf_text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()
        prediction = pdf_text
    elif doc.startswith(b'\x50\x4B\x03\x04'):
        # DOCX file
      

        doc = docx.Document(BytesIO(doc))
        prediction = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    else:
        # Plain text file
        prediction = doc.decode('utf-8')

    if is_arabic(prediction):
        x2 = tokenize_data(arabic_tokenizer, [prediction])

        # Make a prediction
        prediction_input = np.array(x2).reshape(-1)
        prediction_input = pad_sequences([prediction_input], 128)

        output = model1.predict(prediction_input)

        # Update predicted_class based on the output probability
        predicted_class = 'human written' if output > 0.5 else 'machine generated'
    elif is_english(prediction):
        x2 = tokenize_data(tokenizer, [prediction])

        # Make a prediction
        prediction_input = np.array(x2).reshape(-1)
        prediction_input = pad_sequences([prediction_input], 128)

        output = model.predict(prediction_input)

        # Update predicted_class based on the output probability
        predicted_class = 'human written' if output > 0.5 else 'machine generated'
    else:
        predicted_class = 'unknown language'

    return {'predicted_class': predicted_class, 'output': output.tolist()}
    @app.post('/')
    async def main(request: Request): 
        return await request.json()

